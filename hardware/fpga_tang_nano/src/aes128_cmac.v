// ============================================================================
// aes128_cmac.v  -  AES-128-CMAC (NIST SP 800-38B) generator + tag verifier.
//
// INTERFACE (handshake, no SPI, no MCU dependency)
//   start     : 1-cycle pulse. Ignored while busy=1.
//   key       : 128-bit secret key                       -+
//   msg_data  : message, FIRST byte in the MOST-significant |  all captured on the
//               byte (bits [MAX_BYTES*8-1 -: 8]). Bytes at   |  cycle start=1 is
//               index >= msg_len are IGNORED (may be junk).  |  seen, so inputs may
//   msg_len   : message length in BYTES, 0 .. MAX_BYTES      |  change afterwards
//   rx_tag    : 128-bit received CMAC tag                   -+
//   busy      : high from accepted start until done
//   done      : 1-cycle pulse when the result below is valid
//   computed_tag : CMAC(key, message)   (valid at/after done, held until next start)
//   tag_valid    : 1 when computed_tag == rx_tag (valid at/after done)
//   len_err      : 1 if msg_len > MAX_BYTES (nothing computed, tag_valid=0)
//
// WHY A LENGTH-IN-BYTES + FLAT BUFFER INTERFACE
//   CMAC treats a complete last block differently from an incomplete one
//   (K1 vs K2, no padding vs 10* padding), so the hardware must know the exact
//   length. Latching the whole message at 'start' keeps the interface a simple
//   one-shot handshake, which is exactly what a Phase-1 unpacker produces
//   (one 128-bit frame = msg_data[MAX_BYTES*8-1 -: 128], msg_len = 16).
//   Battery/BMS frames are byte-oriented, so the length granularity is bytes.
//
// ALGORITHM (SP 800-38B section 6)
//   L  = AES_K(0^128)
//   K1 = dbl(L)         K2 = dbl(K1)      dbl(x) = (x<<1) ^ (x[127] ? Rb : 0), Rb=0x87
//   n  = max(1, ceil(len/16));   last block complete iff (len>0 && len%16==0)
//   X  = 0
//   for i in 0..n-2 : X = AES_K(X ^ M_i)
//   last = complete ? M_n ^ K1 : (M_n || 1 0*) ^ K2
//   T  = AES_K(X ^ last)
//
// PARAMETER
//   MAX_BYTES : maximum message size, must be a multiple of 16 (default 64).
//
// LATENCY  ~ 12 clocks per AES call, (1 + n) AES calls -> e.g. 16-byte frame ~ 25 clocks
// ============================================================================
module aes128_cmac #(
    parameter MAX_BYTES = 64
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     start,
    input  wire [127:0]             key,
    input  wire [MAX_BYTES*8-1:0]   msg_data,
    input  wire [$clog2(MAX_BYTES+1)-1:0] msg_len,
    input  wire [127:0]             rx_tag,

    output reg                      busy,
    output reg                      done,
    output reg  [127:0]             computed_tag,
    output reg                      tag_valid,
    output reg                      len_err
);
    localparam MAX_BITS = MAX_BYTES * 8;
    localparam LEN_W    = $clog2(MAX_BYTES + 1);
    localparam BLK_W    = LEN_W - 4;               // enough for block index 0..MAX_BYTES/16-1

    localparam [127:0] RB = 128'h00000000000000000000000000000087;

    // ---------------- FSM states ----------------
    localparam [2:0] S_IDLE        = 3'd0,
                     S_SUBKEY      = 3'd1,   // launch AES(K, 0)
                     S_SUBKEY_WAIT = 3'd2,   // wait, then derive K1,K2
                     S_BLOCK       = 3'd3,   // launch AES(K, X ^ block)
                     S_BLOCK_WAIT  = 3'd4;   // wait, then next block / finish

    reg [2:0] state;

    // ---------------- captured operands ----------------
    reg [127:0]       key_reg;
    reg [127:0]       rx_tag_reg;
    reg [MAX_BITS-1:0] msg_buf;      // shifted left by 128 after every block
    reg [BLK_W-1:0]   blk_idx;       // block currently being processed
    reg [BLK_W-1:0]   blk_last;      // index of the final block
    reg               complete;      // final block is a full 16 bytes (len>0, len%16==0)
    reg [3:0]         rem;           // bytes in final block if incomplete (= len % 16)

    // ---------------- CMAC working values ----------------
    reg [127:0] x_reg;               // running CBC-MAC value
    reg [127:0] k1_reg, k2_reg;      // subkeys

    // ---------------- AES core ----------------
    wire         aes_start;
    wire [127:0] aes_in;
    wire         aes_busy, aes_done;
    wire [127:0] aes_out;

    aes128_core u_aes (
        .clk(clk), .rst_n(rst_n),
        .start(aes_start), .key(key_reg), .block_in(aes_in),
        .busy(aes_busy), .done(aes_done), .block_out(aes_out)
    );

    // ---------------- helpers ----------------
    // dbl(): left shift by 1, conditionally XOR Rb (MSB of input decides)
    function [127:0] dbl;
        input [127:0] v;
        begin
            dbl = {v[126:0], 1'b0} ^ (v[127] ? RB : 128'd0);
        end
    endfunction

    // 10* padding: keep bytes 0..r-1, byte r = 0x80, rest 0x00.  (r = 0..15)
    // Byte i lives at bits [127-8i -: 8]  (first byte = MSB).
    function [127:0] pad_block;
        input [127:0] blk;
        input [3:0]   r;
        integer i;
        integer ri;
        begin
            ri = {28'd0, r};
            for (i = 0; i < 16; i = i + 1) begin
                if (i < ri)       pad_block[127-8*i -: 8] = blk[127-8*i -: 8];
                else if (i == ri) pad_block[127-8*i -: 8] = 8'h80;
                else              pad_block[127-8*i -: 8] = 8'h00;
            end
        end
    endfunction

    // ---------------- block selection ----------------
    wire [127:0] cur_block = msg_buf[MAX_BITS-1 -: 128];
    wire         is_last   = (blk_idx == blk_last);

    wire [127:0] last_blk  = complete ? (cur_block ^ k1_reg)
                                      : (pad_block(cur_block, rem) ^ k2_reg);
    wire [127:0] blk_val   = is_last ? last_blk : cur_block;

    assign aes_start = (state == S_SUBKEY) || (state == S_BLOCK);
    assign aes_in    = (state == S_SUBKEY) ? 128'd0 : (x_reg ^ blk_val);

    // ---------------- start-of-operation length decode ----------------
    wire [LEN_W-1:0] len_m1 = msg_len - 1'b1;         // only used when msg_len != 0

    // ---------------- FSM ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            computed_tag <= 128'd0;
            tag_valid    <= 1'b0;
            len_err      <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
            // ------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    computed_tag <= 128'd0;
                    tag_valid    <= 1'b0;
                    if (msg_len > MAX_BYTES) begin
                        len_err <= 1'b1;               // reject, nothing computed
                        done    <= 1'b1;
                    end else begin
                        len_err    <= 1'b0;
                        key_reg    <= key;
                        rx_tag_reg <= rx_tag;
                        msg_buf    <= msg_data;
                        x_reg      <= 128'd0;
                        blk_idx    <= {BLK_W{1'b0}};
                        if (msg_len == 0) begin        // empty message: one padded block
                            blk_last <= {BLK_W{1'b0}};
                            complete <= 1'b0;
                            rem      <= 4'd0;
                        end else begin
                            blk_last <= len_m1[LEN_W-1:4];       // ceil(len/16) - 1
                            complete <= (msg_len[3:0] == 4'd0);
                            rem      <= msg_len[3:0];
                        end
                        busy  <= 1'b1;
                        state <= S_SUBKEY;
                    end
                end
            end
            // ------------------------------------------------------------
            S_SUBKEY: begin                             // aes_start=1, aes_in=0
                state <= S_SUBKEY_WAIT;
            end
            S_SUBKEY_WAIT: begin
                if (aes_done) begin                     // aes_out = L
                    k1_reg <= dbl(aes_out);
                    k2_reg <= dbl(dbl(aes_out));
                    state  <= S_BLOCK;
                end
            end
            // ------------------------------------------------------------
            S_BLOCK: begin                              // aes_start=1
                state <= S_BLOCK_WAIT;
            end
            S_BLOCK_WAIT: begin
                if (aes_done) begin
                    x_reg <= aes_out;
                    if (is_last) begin
                        computed_tag <= aes_out;
                        tag_valid    <= (aes_out == rx_tag_reg);
                        done         <= 1'b1;
                        busy         <= 1'b0;
                        state        <= S_IDLE;
                    end else begin
                        blk_idx <= blk_idx + 1'b1;
                        msg_buf <= msg_buf << 128;
                        state   <= S_BLOCK;
                    end
                end
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
