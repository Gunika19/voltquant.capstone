// ============================================================================
// spi_frame_slave.v  -  SPI slave (mode 0) that receives one alert frame + tag
//
// Bus settings the ESP32 (master) must use:
//     SPI mode 0 (CPOL = 0, CPHA = 0), MSB first, CS active low.
//     SCLK up to ~2 MHz is safe at the 27 MHz FPGA clock (use 1 MHz to start).
//
// ONE TRANSACTION = CS low for 34 bytes (272 clocks), then CS high:
//     bytes  0..15  frame (128 bit, byte 0 = frame[127:120])        master -> FPGA
//     bytes 16..31  CMAC tag (128 bit, byte 16 = tag[127:120])      master -> FPGA
//     byte  32      dummy (send anything; FPGA returns 0x00)
//                   -> gives the pipeline ~8 SCLK periods to finish checking
//     byte  33      STATUS, returned by the FPGA on MISO (send anything)
//
// STATUS byte:   [7]   valid   1 = a verdict for THIS frame is ready
//                [6]   accepted
//                [5:3] reject code (0 none, 1 replay, 2 bad tag, 3 reserved bits,
//                                   4 illegal class)
//                [2:1] class of the last ACCEPTED frame (0 normal, 1 warning,
//                                   2 degraded)
//                [0]   frame dropped (pipeline was busy)
//
// frame_valid pulses for ONE clk right after the 256th data bit, with
// frame_out / tag_out already stable. A transaction cut short (CS rises
// before 256 bits) delivers nothing.
// ============================================================================
module spi_frame_slave (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         spi_sclk,
    input  wire         spi_mosi,
    input  wire         spi_cs_n,
    output reg          spi_miso = 1'b0,

    output reg          frame_valid = 1'b0,
    output reg  [127:0] frame_out   = 128'd0,
    output reg  [127:0] tag_out     = 128'd0,
    input  wire [7:0]   status_in
);

    // ---- synchronise the three asynchronous SPI inputs (2 flops each) ----
    reg [2:0] sclk_s = 3'b000;
    reg [2:0] cs_s   = 3'b111;
    reg [1:0] mosi_s = 2'b00;

    wire cs_active  = ~cs_s[1];
    wire sclk_rise  = (sclk_s[2:1] == 2'b01);
    wire sclk_fall  = (sclk_s[2:1] == 2'b10);

    reg [255:0] sh     = 256'd0;
    reg [8:0]   bitcnt = 9'd0;       // SCLK rising edges seen in this transaction
    reg [7:0]   st_lat = 8'd0;

    wire [2:0]  sidx = 3'd7 - bitcnt[2:0];   // valid while bitcnt is 264..271

    always @(posedge clk) begin
        sclk_s      <= {sclk_s[1:0], spi_sclk};
        cs_s        <= {cs_s[1:0],   spi_cs_n};
        mosi_s      <= {mosi_s[0],   spi_mosi};
        frame_valid <= 1'b0;

        if (!rst_n || !cs_active) begin
            bitcnt   <= 9'd0;
            spi_miso <= 1'b0;
        end else begin
            if (sclk_rise) begin
                if (bitcnt < 9'd256)
                    sh <= {sh[254:0], mosi_s[1]};
                if (bitcnt == 9'd255) begin           // 256th data bit
                    frame_out   <= sh[254:127];
                    tag_out     <= {sh[126:0], mosi_s[1]};
                    frame_valid <= 1'b1;
                end
                if (bitcnt != 9'd511)
                    bitcnt <= bitcnt + 9'd1;
            end

            if (sclk_fall) begin                      // MISO changes on falling edge
                if (bitcnt == 9'd264) begin
                    spi_miso <= status_in[7];
                    st_lat   <= status_in;
                end else if (bitcnt > 9'd264 && bitcnt <= 9'd271) begin
                    spi_miso <= st_lat[sidx];
                end else begin
                    spi_miso <= 1'b0;
                end
            end
        end
    end
endmodule
