`include "alert_frame_defs.vh"

module seq_counter_guard #(
   parameter MAX_CELLS = 64,
   parameter CELL_IDX_W = 6
)(
   input wire           clk,
   input wire           rst_n,
   input wire           valid_in,
   input wire [`CELL_ID_W-1:0] cell_id,
   input wire [`SEQ_W-1:0]      seq_num,
   input wire           commit,
   output reg            valid_out,
   output reg            fresh,
   output reg [2:0]        reject_code,
   output reg [`SEQ_W-1:0]       last_seq_seen
);

  reg [`SEQ_W-1:0] seq_store [0:MAX_CELLS-1];
  reg       seen     [0:MAX_CELLS-1];

  wire [CELL_IDX_W-1:0] idx     = cell_id[CELL_IDX_W-1:0];
  wire [`SEQ_W-1:0] stored = seq_store[idx];
  wire          have_base = seen[idx];
  wire          is_fresh = (!have_base) || (seq_num > stored);

  integer i;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
        fresh         <= 1'b0;
        reject_code <= `REJ_NONE;
        last_seq_seen <= {`SEQ_W{1'b0}};
        for (i = 0; i < MAX_CELLS; i = i + 1) begin
            seq_store[i] <= {`SEQ_W{1'b0}};
            seen[i]     <= 1'b0;
        end
    end else begin
        valid_out <= valid_in;
        if (valid_in) begin
            last_seq_seen <= stored;
            if (is_fresh) begin
                fresh     <= 1'b1;
                reject_code <= `REJ_NONE;
            end else begin
                fresh     <= 1'b0;
                reject_code <= `REJ_REPLAY;
            end
        end else begin
            fresh      <= 1'b0;
            reject_code <= `REJ_NONE;
        end
        if (commit) begin
            seq_store[idx] <= seq_num;
            seen[idx]      <= 1'b1;
        end
    end
  end
endmodule
