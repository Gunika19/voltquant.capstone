`include "alert_frame_defs.vh"

// ============================================================================
// voltquant_alert_top.v  -  Integrated alert path: Phase 1 + Phase 4 + Phase 2
//
//                     +--> frame_unpacker --> seq_counter_guard --+
//   frame_in ---------+                                           |  (results HELD)
//   rx_tag   ---+     +--> aes128_cmac (~25 clocks) --------------+--> alert_fsm
//   aes_key  ---+         tag_valid / done                              (verdict when
//                                                                        CMAC is done)
//
// Unchanged existing modules: frame_unpacker, seq_counter_guard, alert_fsm.
//
// HOW THE MULTI-CYCLE DELAY IS HANDLED
//   * Unpacker (1 clk) and guard (2 clk) finish long before CMAC (~25 clk).
//     Their one-cycle results (struct_ok, fresh, reject codes) are captured
//     into hold registers as they appear.
//   * The verdict is issued ONLY when CMAC pulses done:
//         verdict_valid = cmac_done      tag_ok = cmac tag_valid
//     At that moment the hold registers, the unpacker's held fields and the
//     CMAC result all belong to the SAME frame.
//   * Only ONE frame is in flight at a time. While busy, new frames are
//     ignored (frame_dropped pulses). This is what guarantees the pairing:
//     the unpacker's cell_id/seq_num/class stay stable until the guard has
//     committed, because no new frame can be accepted before then.
//
// FRAME/TAG TIMING: frame_in, rx_tag and aes_key must be valid in the cycle
// frame_valid is high (they are captured on that clock edge).
// CMAC message = the 16 frame bytes, byte 0 = frame_in[127:120] (seq_num MSB).
// ============================================================================
module voltquant_alert_top #(
    parameter SLOW_DIV   = 50_000_000,
    parameter FAST_DIV   = 12_500_000,
    parameter FAULT_HOLD = 100_000_000
) (
    input  wire                      clk,
    input  wire                      rst_n,

    // ---- frame input (later driven by the Phase 3 SPI slave) ----
    input  wire                      frame_valid,
    input  wire [`FRAME_W-1:0]       frame_in,
    input  wire [`TAG_W-1:0]         rx_tag,
    input  wire [127:0]              aes_key,

    // ---- status ----
    output wire                      busy,            // a frame is being processed
    output reg                       frame_dropped,   // 1-clk pulse: frame arrived while busy

    // ---- verdict (from alert_fsm) ----
    output wire                      accepted,
    output wire                      rejected,
    output wire [2:0]                reject_code,
    output wire [`PRED_CLASS_W-1:0]  latched_class,
    output wire [`CELL_ID_W-1:0]     latched_cell,
    output wire [`SEQ_W-1:0]         latched_seq,
    output wire                      led_normal,
    output wire                      led_warning,
    output wire                      led_degraded,
    output wire                      buzzer,
    output wire                      led_fault,
    output wire                      integrity_disabled
);

    // ------------------------------------------------------------------
    // Single-frame-in-flight control
    // ------------------------------------------------------------------
    reg  pipe_busy;
    reg  verdict_d;                         // verdict_valid delayed 1 clk (= commit_seq clk)
    wire cmac_done;
    wire accept_frame = frame_valid & ~pipe_busy;

    assign busy = pipe_busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_busy     <= 1'b0;
            verdict_d     <= 1'b0;
            frame_dropped <= 1'b0;
        end else begin
            verdict_d     <= cmac_done;
            frame_dropped <= frame_valid & pipe_busy;
            if (accept_frame)
                pipe_busy <= 1'b1;
            else if (verdict_d)             // commit_seq is high in this same clk
                pipe_busy <= 1'b0;          // -> free again the clk after the commit
        end
    end

    // ------------------------------------------------------------------
    // Phase 1: unpacker + sequence guard (unchanged modules)
    // ------------------------------------------------------------------
    wire                     u_valid, u_struct_ok;
    wire [`SEQ_W-1:0]        u_seq;
    wire [`CELL_ID_W-1:0]    u_cell;
    wire [`PRED_CLASS_W-1:0] u_class;
    wire [`SOH_W-1:0] u_soh;
    wire [`TIMESTAMP_W-1:0]  u_time;
    wire [2:0]               u_struct_rej;

    frame_unpacker u_unpack (
        .clk(clk), .rst_n(rst_n),
        .valid_in(accept_frame), .frame_in(frame_in),
        .valid_out(u_valid), .seq_num(u_seq), .cell_id(u_cell),
        .pred_class(u_class), .soh(u_soh), .timestamp(u_time),
        .struct_ok(u_struct_ok), .reject_code(u_struct_rej)
    );

    wire                     g_valid, g_fresh, commit_seq;
    wire [2:0]               g_rej;
    wire [`SEQ_W-1:0]        g_last;

    seq_counter_guard #(.MAX_CELLS(64), .CELL_IDX_W(6)) u_guard (
        .clk(clk), .rst_n(rst_n),
        .valid_in(u_valid), .cell_id(u_cell), .seq_num(u_seq),
        .commit(commit_seq),
        .valid_out(g_valid), .fresh(g_fresh),
        .reject_code(g_rej), .last_seq_seen(g_last)
    );

    // ------------------------------------------------------------------
    // Hold registers: struct_ok / fresh (and their codes) are only valid for
    // ONE clock at the output of Phase 1, so keep them until CMAC finishes.
    // (u_seq, u_cell, u_class are already held by the unpacker itself.)
    // ------------------------------------------------------------------
    reg       h_struct_ok, h_fresh;
    reg [2:0] h_struct_rej, h_fresh_rej;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_struct_ok  <= 1'b0;
            h_struct_rej <= `REJ_NONE;
            h_fresh      <= 1'b0;
            h_fresh_rej  <= `REJ_NONE;
        end else begin
            if (u_valid) begin
                h_struct_ok  <= u_struct_ok;
                h_struct_rej <= u_struct_rej;
            end
            if (g_valid) begin
                h_fresh     <= g_fresh;
                h_fresh_rej <= g_rej;
            end
        end
    end

    // ------------------------------------------------------------------
    // Phase 4: AES-128-CMAC over the 16-byte frame, started with the frame
    // ------------------------------------------------------------------
    wire         cmac_busy, cmac_tag_valid, cmac_len_err;
    wire [127:0] cmac_tag;

    aes128_cmac #(.MAX_BYTES(16)) u_cmac (
        .clk(clk), .rst_n(rst_n),
        .start(accept_frame),
        .key(aes_key),
        .msg_data(frame_in),
        .msg_len(5'd16),
        .rx_tag(rx_tag),
        .busy(cmac_busy),
        .done(cmac_done),
        .computed_tag(cmac_tag),
        .tag_valid(cmac_tag_valid),
        .len_err(cmac_len_err)
    );

    // ------------------------------------------------------------------
    // Phase 2: alert FSM (unchanged module). Verdict = CMAC done.
    // ------------------------------------------------------------------
    alert_fsm #(
        .SLOW_DIV(SLOW_DIV), .FAST_DIV(FAST_DIV), .FAULT_HOLD(FAULT_HOLD)
    ) u_fsm (
        .clk(clk), .rst_n(rst_n),
        .verdict_valid(cmac_done),
        .struct_ok(h_struct_ok),   .struct_reject(h_struct_rej),
        .fresh(h_fresh),           .fresh_reject(h_fresh_rej),
        .tag_ok(cmac_tag_valid),   .tag_present(1'b1),
        .pred_class(u_class), .cell_id(u_cell), .seq_num(u_seq),
        .accepted(accepted), .rejected(rejected),
        .reject_code(reject_code), .commit_seq(commit_seq),
        .latched_class(latched_class), .latched_cell(latched_cell),
        .latched_seq(latched_seq),
        .led_normal(led_normal), .led_warning(led_warning),
        .led_degraded(led_degraded), .buzzer(buzzer),
        .led_fault(led_fault), .integrity_disabled(integrity_disabled)
    );

endmodule
