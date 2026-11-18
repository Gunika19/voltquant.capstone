`include "alert_frame_defs.vh"

// ============================================================================
// voltquant_spi_top.v  -  Tang Nano 20K top: ESP32 (SPI master) -> FPGA verifier
//
//   ESP32 --SPI--> spi_frame_slave --> voltquant_alert_top (unchanged)
//                                       unpacker + replay guard + AES-CMAC check
//                                       + alert FSM
//
// The FPGA only VERIFIES: the sender (ESP32 or laptop) computes the CMAC tag
// with the shared key; the key is a constant inside this bitstream.
//
// Pins: clk, SPI (4), 6 LEDs. LED map is the same as the button demo:
//   LED0 NORMAL (steady)   LED1 WARNING (1 Hz)   LED2 DEGRADED (4 Hz)
//   LED3 rejected: bad tag (1.5 s)   LED4 rejected: replay (1.5 s)
//   LED5 accepted flash (1 s)
//
// KEY below is the public NIST test key: fine for bring-up, replace it before
// presenting this as a "secret" (the ESP32/laptop must use the same value).
// ============================================================================
module voltquant_spi_top #(
    parameter CLK_HZ         = 27_000_000,
    parameter LED_ACTIVE_LOW = 1
) (
    input  wire       clk,

    input  wire       spi_sclk,
    input  wire       spi_mosi,
    input  wire       spi_cs_n,
    output wire       spi_miso,

    output wire [5:0] led
);

    localparam [127:0] KEY = 128'h2b7e151628aed2a6abf7158809cf4f3c;

    // ---------------- power-on reset ----------------
    reg [7:0] rst_cnt = 8'd0;
    wire      rst_n   = (rst_cnt == 8'hFF);
    always @(posedge clk) if (!rst_n) rst_cnt <= rst_cnt + 8'd1;

    // ---------------- SPI slave ----------------
    wire         sp_fv;
    wire [127:0] sp_frame, sp_tag;
    wire [7:0]   status;

    spi_frame_slave u_spi (
        .clk(clk), .rst_n(rst_n),
        .spi_sclk(spi_sclk), .spi_mosi(spi_mosi),
        .spi_cs_n(spi_cs_n), .spi_miso(spi_miso),
        .frame_valid(sp_fv), .frame_out(sp_frame), .tag_out(sp_tag),
        .status_in(status)
    );

    // ---------------- the real pipeline ----------------
    wire       led_normal, led_warning, led_degraded, buzzer, led_fault;
    wire       accepted, rejected, dropped;
    wire [2:0] reject_code;
    wire [`PRED_CLASS_W-1:0] latched_class;

    voltquant_alert_top #(
        .SLOW_DIV   (CLK_HZ / 2),
        .FAST_DIV   (CLK_HZ / 8),
        .FAULT_HOLD (CLK_HZ + CLK_HZ / 2)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .frame_valid(sp_fv), .frame_in(sp_frame),
        .rx_tag(sp_tag), .aes_key(KEY),
        .busy(), .frame_dropped(dropped),
        .accepted(accepted), .rejected(rejected), .reject_code(reject_code),
        .latched_class(latched_class), .latched_cell(), .latched_seq(),
        .led_normal(led_normal), .led_warning(led_warning),
        .led_degraded(led_degraded), .buzzer(buzzer),
        .led_fault(led_fault), .integrity_disabled()
    );

    // ---------------- status byte returned to the ESP32 ----------------
    reg       res_valid = 1'b0, res_acc = 1'b0, res_drop = 1'b0;
    reg [2:0] res_code  = 3'd0;

    always @(posedge clk) begin
        if (sp_fv) begin res_valid <= 1'b0; res_drop <= 1'b0; end
        if (accepted) begin res_valid <= 1'b1; res_acc <= 1'b1; res_code <= 3'd0; end
        if (rejected) begin res_valid <= 1'b1; res_acc <= 1'b0; res_code <= reject_code; end
        if (dropped)  res_drop <= 1'b1;
    end

    assign status = { res_valid, res_acc, res_code, latched_class, res_drop };

    // ---------------- LEDs ----------------
    reg [31:0] af = 32'd0;
    always @(posedge clk) begin
        if (accepted)     af <= CLK_HZ;
        else if (af != 0) af <= af - 32'd1;
    end

    wire tag_fault    = led_fault && (reject_code == `REJ_TAG);
    wire replay_fault = led_fault && (reject_code == `REJ_REPLAY);

    wire [5:0] led_pos = { (af != 0), replay_fault, tag_fault,
                           led_degraded, led_warning, led_normal };

    assign led = LED_ACTIVE_LOW ? ~led_pos : led_pos;

endmodule
