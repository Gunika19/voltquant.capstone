`include "alert_frame_defs.vh"

// ============================================================================
// voltquant_board_top.v  -  Button-driven demo wrapper for Tang Nano 20K
//
// Wraps the unchanged voltquant_alert_top. Pins used: 27 MHz clock, the two
// onboard buttons S1/S2, and the 6 onboard LEDs. The frame, key and CMAC tag
// are generated INSIDE the FPGA from constants (no SPI yet).
//
//   S1  = send the next VALID frame        (cycles NORMAL -> WARNING -> DEGRADED
//                                            -> back to NORMAL with a fresh start)
//   S2  = ATTACK on the last valid frame   (first press = one bit flipped, then
//                                            replay, then bit flip, replay, ...)
//
//   LED0  NORMAL    steady        (class of the last ACCEPTED frame)
//   LED1  WARNING   blinks 1 Hz
//   LED2  DEGRADED  blinks 4 Hz
//   LED3  REJECTED: bad tag (tampered frame)          ~1.5 s
//   LED4  REJECTED: replayed frame                    ~1.5 s
//   LED5  ACCEPTED flash                              ~1 s
//
// Rejected frames never change LED0-2: the system keeps showing the last
// trusted class. Button idle level is detected automatically at power-up, so
// active-high or active-low buttons both work (do not hold a button while
// plugging in). Key = public NIST test key; tags computed with Python
// `cryptography`, not by this RTL.
// ============================================================================

// ---- debounced one-clock "press" pulse; idle level auto-captured on arm ----
module vq_btn #(
    parameter DEBOUNCE = 540_000            // 20 ms at 27 MHz
) (
    input  wire clk,
    input  wire din,
    input  wire arm,                        // 1-clk pulse: capture idle level
    output reg  press = 1'b0
);
    reg        s0 = 1'b0, s1 = 1'b0;
    reg        idle = 1'b0, armed = 1'b0, stable = 1'b0;
    reg [31:0] cnt = 32'd0;
    wire       raw = armed & (s1 != idle);

    always @(posedge clk) begin
        s0    <= din;
        s1    <= s0;
        press <= 1'b0;
        if (arm) begin
            idle   <= s1;
            armed  <= 1'b1;
            stable <= 1'b0;
            cnt    <= 32'd0;
        end else if (raw == stable) begin
            cnt <= 32'd0;
        end else if (cnt == DEBOUNCE - 1) begin
            stable <= raw;
            cnt    <= 32'd0;
            if (raw) press <= 1'b1;
        end else begin
            cnt <= cnt + 32'd1;
        end
    end
endmodule

module voltquant_board_top #(
    parameter CLK_HZ         = 27_000_000,
    parameter DEBOUNCE       = 540_000,      // 20 ms
    parameter LOCKOUT        = 2_700_000,    // 100 ms between actions
    parameter LED_ACTIVE_LOW = 1             // Tang Nano 20K LEDs: 0 = ON
) (
    input  wire       clk,
    input  wire       s1,                    // onboard button S1
    input  wire       s2,                    // onboard button S2
    output wire [5:0] led
);

    // ---------------- constants ----------------
    localparam [127:0] KEY = 128'h2b7e151628aed2a6abf7158809cf4f3c;

    localparam [127:0] V0 = 128'h000000010532000000fa000000000000;  // seq1 NORMAL
    localparam [127:0] V1 = 128'h00000002056d000001f4000000000000;  // seq2 WARNING
    localparam [127:0] V2 = 128'h0000000305a3000002ee000000000000;  // seq3 DEGRADED

    localparam [127:0] T0 = 128'hd2384a0ab07031008a5ec3b5d8694578;
    localparam [127:0] T1 = 128'h2f56fe944a16374fa10c8dba53ccfb4d;
    localparam [127:0] T2 = 128'hecad2202e585ecb31ffed3c26a170e40;

    localparam [127:0] FLIP = 128'd1 << 80;   // one bit inside the confidence field

    // ---------------- power-on reset ----------------
    reg [7:0] rst_cnt  = 8'd0;
    wire      por_done = (rst_cnt == 8'hFF);
    reg       por_d    = 1'b0;
    wire      arm      = por_done & ~por_d;   // one pulse when POR ends

    always @(posedge clk) begin
        por_d <= por_done;
        if (!por_done) rst_cnt <= rst_cnt + 8'd1;
    end

    // ---------------- buttons ----------------
    wire p1, p2;
    vq_btn #(.DEBOUNCE(DEBOUNCE)) u_b1 (.clk(clk), .din(s1), .arm(arm), .press(p1));
    vq_btn #(.DEBOUNCE(DEBOUNCE)) u_b2 (.clk(clk), .din(s2), .arm(arm), .press(p2));

    // ---------------- action sequencer ----------------
    localparam [1:0] ST_IDLE = 2'd0, ST_RST = 2'd1, ST_SEND = 2'd2, ST_LOCK = 2'd3;

    reg [1:0]  st       = ST_IDLE;
    reg [31:0] cnt      = 32'd0;
    reg        rst_n_r  = 1'b0;
    reg        fv       = 1'b0;
    reg [1:0]  vi       = 2'd0;      // next valid frame index
    reg [1:0]  last     = 2'd0;      // last accepted-frame index
    reg [1:0]  use_idx  = 2'd0;      // frame being sent
    reg        flip     = 1'b0;      // flip a payload bit while sending
    reg        is_valid = 1'b0;      // current action is a valid send
    reg        sent_any = 1'b0;
    reg        need_rst = 1'b0;      // wrapped: reset guard before next valid frame
    reg        atk_tgl  = 1'b0;      // 0 = tamper next, 1 = replay next

    always @(posedge clk) begin
        fv <= 1'b0;
        if (!por_done) begin
            rst_n_r  <= 1'b0;
            st       <= ST_IDLE;
            cnt      <= 32'd0;
            vi       <= 2'd0;
            sent_any <= 1'b0;
            need_rst <= 1'b0;
            atk_tgl  <= 1'b0;
        end else begin
            case (st)
              ST_IDLE: begin
                  rst_n_r <= 1'b1;
                  cnt     <= 32'd0;
                  if (p1) begin
                      use_idx  <= vi;
                      flip     <= 1'b0;
                      is_valid <= 1'b1;
                      st       <= need_rst ? ST_RST : ST_SEND;
                      need_rst <= 1'b0;
                  end else if (p2 && sent_any) begin
                      use_idx  <= last;
                      flip     <= ~atk_tgl;     // tamper first, then replay
                      atk_tgl  <= ~atk_tgl;
                      is_valid <= 1'b0;
                      st       <= ST_SEND;
                  end
              end
              ST_RST: begin                      // clear guard memory + latched class
                  cnt <= cnt + 32'd1;
                  rst_n_r <= (cnt >= 32'd16);
                  if (cnt == 32'd32) begin
                      cnt <= 32'd0;
                      st  <= ST_SEND;
                  end
              end
              ST_SEND: begin
                  fv  <= 1'b1;
                  cnt <= 32'd0;
                  st  <= ST_LOCK;
                  if (is_valid) begin
                      last     <= use_idx;
                      sent_any <= 1'b1;
                      atk_tgl  <= 1'b0;
                      vi       <= (use_idx == 2'd2) ? 2'd0 : use_idx + 2'd1;
                      need_rst <= (use_idx == 2'd2);
                  end
              end
              ST_LOCK: begin
                  if (cnt == LOCKOUT - 1) begin
                      cnt <= 32'd0;
                      st  <= ST_IDLE;
                  end else
                      cnt <= cnt + 32'd1;
              end
            endcase
        end
    end

    // frame + tag currently selected (stable while a send is in progress)
    reg [127:0] base_frame, tag_sel;
    always @(*) begin
        case (use_idx)
            2'd0:    begin base_frame = V0; tag_sel = T0; end
            2'd1:    begin base_frame = V1; tag_sel = T1; end
            default: begin base_frame = V2; tag_sel = T2; end
        endcase
    end
    wire [127:0] frame_sel = flip ? (base_frame ^ FLIP) : base_frame;

    // ---------------- the real pipeline ----------------
    wire       led_normal, led_warning, led_degraded, buzzer, led_fault, accepted;
    wire [2:0] reject_code;

    voltquant_alert_top #(
        .SLOW_DIV   (CLK_HZ / 2),          // 1 Hz blink (warning)
        .FAST_DIV   (CLK_HZ / 8),          // 4 Hz blink (degraded)
        .FAULT_HOLD (CLK_HZ + CLK_HZ / 2)  // 1.5 s fault indication
    ) u_core (
        .clk(clk), .rst_n(rst_n_r),
        .frame_valid(fv), .frame_in(frame_sel),
        .rx_tag(tag_sel), .aes_key(KEY),
        .busy(), .frame_dropped(),
        .accepted(accepted), .rejected(), .reject_code(reject_code),
        .latched_class(), .latched_cell(), .latched_seq(),
        .led_normal(led_normal), .led_warning(led_warning),
        .led_degraded(led_degraded), .buzzer(buzzer),
        .led_fault(led_fault), .integrity_disabled()
    );

    // ---------------- LEDs ----------------
    reg [31:0] af = 32'd0;                       // accepted-flash timer
    always @(posedge clk) begin
        if (accepted)      af <= CLK_HZ;
        else if (af != 0)  af <= af - 32'd1;
    end

    wire tag_fault    = led_fault && (reject_code == `REJ_TAG);
    wire replay_fault = led_fault && (reject_code == `REJ_REPLAY);

    wire [5:0] led_pos = { (af != 0), replay_fault, tag_fault,
                           led_degraded, led_warning, led_normal };

    assign led = LED_ACTIVE_LOW ? ~led_pos : led_pos;

endmodule