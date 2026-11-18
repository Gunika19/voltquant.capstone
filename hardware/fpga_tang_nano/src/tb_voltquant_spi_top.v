`timescale 1ns/1ps
`include "alert_frame_defs.vh"

// Acts as the ESP32: SPI master, mode 0, MSB first, 34-byte transactions.
// Checks the returned STATUS byte AND the verdicts seen inside the pipeline.
module tb_voltquant_spi_top;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg  sclk = 1'b0, mosi = 1'b0, cs_n = 1'b1;
    wire miso;
    wire [5:0] led;

    voltquant_spi_top #(.CLK_HZ(2000), .LED_ACTIVE_LOW(1)) dut (
        .clk(clk), .spi_sclk(sclk), .spi_mosi(mosi),
        .spi_cs_n(cs_n), .spi_miso(miso), .led(led));

    localparam [127:0] V0 = 128'h000000010532000000fa000000000000;  // seq1 NORMAL
    localparam [127:0] V1 = 128'h00000002056d000001f4000000000000;  // seq2 WARNING
    localparam [127:0] V2 = 128'h0000000305a3000002ee000000000000;  // seq3 DEGRADED
    localparam [127:0] T0 = 128'hd2384a0ab07031008a5ec3b5d8694578;
    localparam [127:0] T1 = 128'h2f56fe944a16374fa10c8dba53ccfb4d;
    localparam [127:0] T2 = 128'hecad2202e585ecb31ffed3c26a170e40;
    localparam [127:0] FLIP = 128'd1 << 80;

    integer pass = 0, fail = 0;
    task check(input ok, input [8*56-1:0] what);
        begin
            if (ok) pass = pass + 1;
            else begin fail = fail + 1; $display("FAIL: %0s  (t=%0t)", what, $time); end
        end
    endtask

    // ---- verdict log from inside the pipeline ----
    integer nev = 0;
    reg       ev_acc   [0:31];
    reg [2:0] ev_code  [0:31];
    reg [1:0] ev_class [0:31];
    always @(posedge clk) begin
        if (dut.u_core.accepted) begin
            ev_acc[nev] = 1'b1; ev_code[nev] = dut.u_core.reject_code;
            ev_class[nev] = dut.u_core.latched_class; nev = nev + 1;
        end
        if (dut.u_core.rejected) begin
            ev_acc[nev] = 1'b0; ev_code[nev] = dut.u_core.reject_code;
            ev_class[nev] = dut.u_core.latched_class; nev = nev + 1;
        end
    end

    // ---- SPI master model ----
    integer half = 14;            // clk cycles per SCLK half period (14 ~ 1 MHz @ 27 MHz)
    reg [7:0] status_rx;
    reg       miso_early;         // MISO must stay 0 before the status byte

    task xfer(input [127:0] f, input [127:0] t, input integer nbits);
        integer i;
        reg [255:0] d;
        begin
            d = {f, t};
            status_rx  = 8'd0;
            miso_early = 1'b0;
            cs_n = 1'b0;
            repeat (6) @(posedge clk);
            for (i = 0; i < nbits; i = i + 1) begin
                mosi = (i < 256) ? d[255 - i] : 1'b0;
                repeat (half) @(posedge clk);
                sclk = 1'b1;                                   // slave samples MOSI
                if (i < 264 && miso === 1'b1) miso_early = 1'b1;
                if (i >= 264 && i < 272) status_rx[271 - i] = miso;
                repeat (half) @(posedge clk);
                sclk = 1'b0;
            end
            repeat (6) @(posedge clk);
            cs_n = 1'b1;
            repeat (20) @(posedge clk);
        end
    endtask

    wire       s_valid = status_rx[7];
    wire       s_acc   = status_rx[6];
    wire [2:0] s_code  = status_rx[5:3];
    wire [1:0] s_class = status_rx[2:1];

    integer n_before;
    initial begin
        repeat (400) @(posedge clk);

        // 1. valid NORMAL frame
        xfer(V0, T0, 272);
        check(s_valid && s_acc && s_code == 3'd0 && s_class == `CLASS_NORMAL, "valid frame: ACCEPT, class NORMAL in status");
        check(!miso_early, "MISO silent before status byte");

        // 2. same frame with one flipped payload bit, original tag
        xfer(V0 ^ FLIP, T0, 272);
        check(s_valid && !s_acc && s_code == `REJ_TAG, "tampered frame: REJECT bad tag in status");
        check(s_class == `CLASS_NORMAL, "tamper does not change trusted class");

        // 3. replay of the accepted frame
        xfer(V0, T0, 272);
        check(s_valid && !s_acc && s_code == `REJ_REPLAY, "replayed frame: REJECT replay in status");

        // 4. next valid frame: WARNING
        xfer(V1, T1, 272);
        check(s_valid && s_acc && s_class == `CLASS_WARNING, "valid frame: ACCEPT, class WARNING");

        // 5. fast SCLK (~2.7 MHz at 27 MHz): DEGRADED
        half = 5;
        xfer(V2, T2, 272);
        check(s_valid && s_acc && s_class == `CLASS_DEGRADED, "fast SCLK: ACCEPT, class DEGRADED");
        half = 14;

        // 6. transaction cut short: no verdict, state intact
        n_before = nev;
        xfer(V2, T2, 100);
        check(nev == n_before, "short transaction delivers nothing");
        xfer(V2 ^ FLIP, T2, 272);
        check(s_valid && !s_acc && s_code == `REJ_TAG, "after aborted transfer: next frame still checked");

        // 7. verdict order inside the pipeline
        check(nev == 6, "exactly 6 verdicts");
        check(ev_acc[0]===1 && ev_class[0]===`CLASS_NORMAL,   "ev0 ACCEPT NORMAL");
        check(ev_acc[1]===0 && ev_code[1]===`REJ_TAG,         "ev1 REJECT tag");
        check(ev_acc[2]===0 && ev_code[2]===`REJ_REPLAY,      "ev2 REJECT replay");
        check(ev_acc[3]===1 && ev_class[3]===`CLASS_WARNING,  "ev3 ACCEPT WARNING");
        check(ev_acc[4]===1 && ev_class[4]===`CLASS_DEGRADED, "ev4 ACCEPT DEGRADED");
        check(ev_acc[5]===0 && ev_code[5]===`REJ_TAG,         "ev5 REJECT tag");

        $display("========================================");
        $display("SPI TOP VERIFICATION COMPLETE");
        $display("Checks passed : %0d", pass);
        $display("Checks failed : %0d", fail);
        if (fail == 0) $display("ALL TESTS PASSED  (%0d checks)", pass);
        $display("========================================");
        $finish;
    end
endmodule
