// ============================================================================
// aes128_core.v  -  Iterative AES-128 ENCRYPT core (FIPS-197), one round/cycle.
//
// Byte ordering (FIPS-197 section 3.4 / 3.5):
//   The 128-bit bus is big-endian: input byte 0 is bits [127:120],
//   byte 15 is bits [7:0].  The AES state is column-major:
//        state[row r][col c] = byte (r + 4*c)
//   so column 0 is bits [127:96], column 3 is bits [31:0].
//
// Timing (per block):
//   cycle 0 : start=1 seen  -> state <= block_in ^ key            (round 0)
//   cycle 1..10 : one round each (round 10 has no MixColumns)
//   then    : done=1 for ONE cycle, block_out valid (and held until next start)
//   => done appears 11 clock edges after the edge that accepted start.
//
// 'start' is only accepted while busy=0. 'key' and 'block_in' must be stable
// during the cycle in which start=1 (they are captured on that edge).
// ============================================================================
module aes128_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [127:0] key,
    input  wire [127:0] block_in,
    output reg          busy,
    output reg          done,
    output wire [127:0] block_out
);
    // ---------------- helpers ----------------
    function [7:0] xtime;               // multiply by {02} in GF(2^8)
        input [7:0] b;
        begin
            xtime = {b[6:0], 1'b0} ^ (b[7] ? 8'h1b : 8'h00);
        end
    endfunction

    // ShiftRows: out[r,c] = in[r, (c+r) mod 4]  ->  byte(r+4c) = in byte(r+4*((c+r)%4))
    function [127:0] shift_rows;
        input [127:0] s;
        integer c, r, src;
        begin
            for (c = 0; c < 4; c = c + 1)
                for (r = 0; r < 4; r = r + 1) begin
                    src = r + 4 * ((c + r) % 4);
                    shift_rows[127 - 8*(r + 4*c) -: 8] = s[127 - 8*src -: 8];
                end
        end
    endfunction

    // MixColumns on one column {s0,s1,s2,s3} (s0 = row 0 = MSB byte)
    function [31:0] mix_col;
        input [31:0] col;
        reg [7:0] s0, s1, s2, s3;
        begin
            s0 = col[31:24]; s1 = col[23:16]; s2 = col[15:8]; s3 = col[7:0];
            mix_col[31:24] = xtime(s0) ^ (xtime(s1) ^ s1) ^ s2 ^ s3;      // 2s0 3s1 1s2 1s3
            mix_col[23:16] = s0 ^ xtime(s1) ^ (xtime(s2) ^ s2) ^ s3;      // 1s0 2s1 3s2 1s3
            mix_col[15:8]  = s0 ^ s1 ^ xtime(s2) ^ (xtime(s3) ^ s3);      // 1s0 1s1 2s2 3s3
            mix_col[7:0]   = (xtime(s0) ^ s0) ^ s1 ^ s2 ^ xtime(s3);      // 3s0 1s1 1s2 2s3
        end
    endfunction

    // ---------------- registers ----------------
    reg [127:0] state;
    reg [127:0] rk;         // current round key
    reg [7:0]   rcon;       // round constant used to derive the NEXT round key
    reg [3:0]   round;      // 1..10 while busy

    assign block_out = state;

    // ---------------- combinational round ----------------
    // SubBytes
    wire [127:0] sub_bytes;
    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : g_sbox
            aes_sbox u_sb (.in(state[127-8*g -: 8]), .out(sub_bytes[127-8*g -: 8]));
        end
    endgenerate

    // ShiftRows, MixColumns
    wire [127:0] shifted = shift_rows(sub_bytes);
    wire [127:0] mixed   = { mix_col(shifted[127:96]), mix_col(shifted[95:64]),
                             mix_col(shifted[63:32]),  mix_col(shifted[31:0]) };

    // Key schedule step (round key for THIS round)
    wire [127:0] next_rk;
    aes_key_expand u_kx (.key_in(rk), .rcon(rcon), .key_out(next_rk));

    wire         last_round = (round == 4'd10);
    wire [127:0] round_out  = (last_round ? shifted : mixed) ^ next_rk;   // AddRoundKey

    // ---------------- control ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy  <= 1'b0;
            done  <= 1'b0;
            round <= 4'd0;
            state <= 128'd0;
            rk    <= 128'd0;
            rcon  <= 8'h01;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                if (start) begin
                    state <= block_in ^ key;        // initial AddRoundKey
                    rk    <= key;
                    rcon  <= 8'h01;
                    round <= 4'd1;
                    busy  <= 1'b1;
                end
            end else begin
                state <= round_out;
                rk    <= next_rk;
                rcon  <= xtime(rcon);               // 01,02,04,08,10,20,40,80,1b,36
                if (last_round) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    round <= round + 4'd1;
                end
            end
        end
    end
endmodule
