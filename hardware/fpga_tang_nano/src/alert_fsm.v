`include "alert_frame_defs.vh"

module alert_fsm #(
    parameter SLOW_DIV   = 50_000_000,   // ~1 Hz at 100 MHz -> warning
    parameter FAST_DIV   = 12_500_000,   // ~4 Hz            -> degraded
    parameter FAULT_HOLD = 100_000_000   // ~1 s fault LED hold
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      verdict_valid,
    input  wire                      struct_ok,
    input  wire [2:0]                struct_reject,
    input  wire                      fresh,
    input  wire [2:0]                fresh_reject,
    input  wire                      tag_ok,
    input  wire                      tag_present,
    input  wire [`PRED_CLASS_W-1:0]  pred_class,
    input  wire [`CELL_ID_W-1:0]     cell_id,
    input  wire [`SEQ_W-1:0]         seq_num,
    output reg                       accepted,
    output reg                       rejected,
    output reg  [2:0]                reject_code,
    output reg                       commit_seq,
    output reg  [`PRED_CLASS_W-1:0]  latched_class,
    output reg  [`CELL_ID_W-1:0]     latched_cell,
    output reg  [`SEQ_W-1:0]         latched_seq,
    output wire                      led_normal,
    output wire                      led_warning,
    output wire                      led_degraded,
    output wire                      buzzer,
    output reg                       led_fault,
    output wire                      integrity_disabled
);

    assign integrity_disabled = ~tag_present;

    wire integrity_ok = (!tag_present) || tag_ok;
    wire all_ok = struct_ok && fresh && integrity_ok;

    reg [31:0] fault_timer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accepted      <= 1'b0;
            rejected      <= 1'b0;
            reject_code   <= `REJ_NONE;
            commit_seq    <= 1'b0;
            latched_class <= `CLASS_NORMAL;
            latched_cell  <= {`CELL_ID_W{1'b0}};
            latched_seq   <= {`SEQ_W{1'b0}};
            led_fault     <= 1'b0;
            fault_timer   <= 32'd0;
        end else begin
            accepted   <= 1'b0;
            rejected   <= 1'b0;
            commit_seq <= 1'b0;

            // Fault-LED hold timer. This block MUST come BEFORE the verdict
            // logic: a reject below sets led_fault <= 1 and reloads the timer,
            // and the last non-blocking assignment in a block wins.
            if (fault_timer != 32'd0)
                fault_timer <= fault_timer - 32'd1;
            else
                led_fault <= 1'b0;

            if (verdict_valid) begin
                if (all_ok) begin
                    accepted      <= 1'b1;
                    commit_seq    <= 1'b1;
                    reject_code   <= `REJ_NONE;
                    latched_class <= pred_class;
                    latched_cell  <= cell_id;
                    latched_seq   <= seq_num;
                end else begin
                    rejected <= 1'b1;

                    // Precedence: structure, integrity, freshness
                    if (!struct_ok)
                        reject_code <= struct_reject;
                    else if (tag_present && !tag_ok)
                        reject_code <= `REJ_TAG;
                    else
                        reject_code <= fresh_reject;

                    led_fault   <= 1'b1;
                    fault_timer <= FAULT_HOLD;
                end
            end

        end
    end

    reg [31:0] slow_cnt;
    reg slow_tick;

    reg [31:0] fast_cnt;
    reg fast_tick;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slow_cnt  <= 32'd0;
            slow_tick <= 1'b0;
            fast_cnt  <= 32'd0;
            fast_tick <= 1'b0;
        end else begin
            if (slow_cnt >= SLOW_DIV - 1) begin
                slow_cnt  <= 32'd0;
                slow_tick <= ~slow_tick;
            end else begin
                slow_cnt <= slow_cnt + 32'd1;
            end

            if (fast_cnt >= FAST_DIV - 1) begin
                fast_cnt  <= 32'd0;
                fast_tick <= ~fast_tick;
            end else begin
                fast_cnt <= fast_cnt + 32'd1;
            end
        end
    end

    assign led_normal   = (latched_class == `CLASS_NORMAL);
    assign led_warning  = (latched_class == `CLASS_WARNING) && slow_tick;
    assign led_degraded = (latched_class == `CLASS_DEGRADED) && fast_tick;
    assign buzzer       = (latched_class == `CLASS_DEGRADED) && fast_tick;

endmodule
