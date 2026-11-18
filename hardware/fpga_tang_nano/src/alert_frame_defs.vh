`ifndef ALERT_FRAME_DEFS_VH
`define ALERT_FRAME_DEFS_VH

`define FRAME_W          128
`define TAG_W          128

`define SEQ_W           32
`define CELL_ID_W         8
`define PRED_CLASS_W          2
`define SOH_W                  8   // state of health, 0..255 = 0..100 %
`define TIMESTAMP_W         32
`define RESERVED_W          46

`define SEQ_HI     127
`define SEQ_LO       96
`define CELL_ID_HI    95
`define CELL_ID_LO     88
`define PRED_CLASS_HI 87
`define PRED_CLASS_LO 86
`define SOH_HI        85
`define SOH_LO        78
`define TIMESTAMP_HI     77
`define TIMESTAMP_LO      46
`define RESERVED_HI      45
`define RESERVED_LO        0

// Bits that MUST be zero. 1 = checked. To give a new field some of the spare
// bits later, clear those bits here (and nothing else in the FPGA changes).
`define RESERVED_CHECK_MASK 46'h3FFF_FFFF_FFFF

// Old names kept so older testbenches still compile (same bits as SOH)
`define CONFIDENCE_W  `SOH_W
`define CONFIDENCE_HI `SOH_HI
`define CONFIDENCE_LO `SOH_LO

`define CLASS_NORMAL 2'd0
`define CLASS_WARNING 2'd1
`define CLASS_DEGRADED 2'd2
`define CLASS_RESERVED 2'd3

`define REJ_NONE    3'd0
`define REJ_REPLAY    3'd1
`define REJ_TAG    3'd2
`define REJ_RESERVED_NZ 3'd3
`define REJ_BAD_CLASS 3'd4

`endif
