`include "alert_frame_defs.vh"

module frame_unpacker (
  input wire             clk,
  input wire             rst_n,
  input wire             valid_in,
  input wire [`FRAME_W-1:0]        frame_in,
  output reg              valid_out,
     output reg [`SEQ_W-1:0]        seq_num,
     output reg [`CELL_ID_W-1:0]      cell_id,
     output reg [`PRED_CLASS_W-1:0] pred_class,
     output reg [`SOH_W-1:0]        soh,
     output reg [`TIMESTAMP_W-1:0]      timestamp,
     output reg              struct_ok,
     output reg [2:0]         reject_code
);

  wire [`SEQ_W-1:0]    w_seq   = frame_in[`SEQ_HI   : `SEQ_LO];
  wire [`CELL_ID_W-1:0] w_cell = frame_in[`CELL_ID_HI : `CELL_ID_LO];
  wire [`PRED_CLASS_W-1:0] w_class = frame_in[`PRED_CLASS_HI : `PRED_CLASS_LO];
  wire [`SOH_W-1:0] w_soh = frame_in[`SOH_HI : `SOH_LO];
  wire [`TIMESTAMP_W-1:0] w_time = frame_in[`TIMESTAMP_HI : `TIMESTAMP_LO];
  wire [`RESERVED_W-1:0] w_reserved = frame_in[`RESERVED_HI : `RESERVED_LO];

     wire reserved_nonzero = |(w_reserved & `RESERVED_CHECK_MASK);
     wire class_illegal = (w_class == `CLASS_RESERVED);

     always @(posedge clk or negedge rst_n) begin
       if (!rst_n) begin
           valid_out <= 1'b0;
           seq_num <= {`SEQ_W{1'b0}};
           cell_id <= {`CELL_ID_W{1'b0}};
           pred_class <= `CLASS_NORMAL;
           soh <= {`SOH_W{1'b0}};
           timestamp <= {`TIMESTAMP_W{1'b0}};
           struct_ok <= 1'b0;
           reject_code <= `REJ_NONE;
       end else begin
           valid_out <= valid_in;
           if (valid_in) begin
               seq_num <= w_seq;
               cell_id <= w_cell;
               pred_class <= w_class;
               soh <= w_soh;
               timestamp <= w_time;
               if (reserved_nonzero) begin
                   struct_ok <= 1'b0;
                   reject_code <= `REJ_RESERVED_NZ;
               end else if (class_illegal) begin
                   struct_ok <= 1'b0;
                   reject_code <= `REJ_BAD_CLASS;
        end else begin
           struct_ok <= 1'b1;
           reject_code <= `REJ_NONE;
        end
      end else begin
        struct_ok <= 1'b0;
        reject_code <= `REJ_NONE;
      end
    end
  end
endmodule
