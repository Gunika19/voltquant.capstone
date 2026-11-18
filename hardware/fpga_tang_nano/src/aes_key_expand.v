// ============================================================================
// aes_key_expand.v  -  ONE step of the AES-128 key schedule (FIPS-197 5.2).
//
// key_in  = round key i   (128 bits, w0 in [127:96] ... w3 in [31:0])
// key_out = round key i+1
//
//   t   = SubWord(RotWord(w3)) ^ {rcon, 24'h0}
//   w0' = w0 ^ t
//   w1' = w1 ^ w0'
//   w2' = w2 ^ w1'
//   w3' = w3 ^ w2'
//
// The AES core calls this once per round ("on the fly"), so only ONE
// 128-bit round key register is needed instead of a stored 11-key table.
// Combinational.
// ============================================================================
module aes_key_expand (
    input  wire [127:0] key_in,
    input  wire [7:0]   rcon,
    output wire [127:0] key_out
);
    wire [31:0] w0 = key_in[127:96];
    wire [31:0] w1 = key_in[95:64];
    wire [31:0] w2 = key_in[63:32];
    wire [31:0] w3 = key_in[31:0];

    // RotWord: [a0 a1 a2 a3] -> [a1 a2 a3 a0]
    wire [31:0] rot = {w3[23:0], w3[31:24]};

    // SubWord
    wire [31:0] sub;
    aes_sbox sb0 (.in(rot[31:24]), .out(sub[31:24]));
    aes_sbox sb1 (.in(rot[23:16]), .out(sub[23:16]));
    aes_sbox sb2 (.in(rot[15:8]),  .out(sub[15:8]));
    aes_sbox sb3 (.in(rot[7:0]),   .out(sub[7:0]));

    wire [31:0] t  = sub ^ {rcon, 24'h000000};
    wire [31:0] n0 = w0 ^ t;
    wire [31:0] n1 = w1 ^ n0;
    wire [31:0] n2 = w2 ^ n1;
    wire [31:0] n3 = w3 ^ n2;

    assign key_out = {n0, n1, n2, n3};
endmodule
