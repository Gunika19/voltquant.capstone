// ============================================================================
// voltquant_esp32.ino  -  ESP32: USB-serial in, AES-CMAC tag, SPI out to FPGA
//
//   model laptop --USB serial--> ESP32 --SPI (mode 0, 1 MHz)--> Tang Nano 20K
//
// LAPTOP -> ESP32 (one text line per prediction, 115200 baud):
//     <class>,<cell_id>[,<soh_percent 0-100>]       class = safe | risk | warning |
//                                                   degraded (or 0 | 1 | 2)
//     examples:   SAFE,5      WARNING,5,87.5      degraded,12,71
//     (SOH is optional; leave it out and the frame carries 0 = not provided)
//   Demo commands:
//     TAMPER <line>   same as <line> but one payload bit is flipped AFTER the
//                     tag is computed (the FPGA must reject it: bad tag)
//     REPLAY          re-sends the previous transaction unchanged (FPGA must
//                     reject it: replay)
//
// ESP32 -> LAPTOP: one "TX ..." line and one "RX ..." line per frame.
//
// Wiring (3.3 V logic, common GND):
//     ESP32 GPIO18 (SCK)  -> FPGA pin 27     ESP32 GPIO5  (CS)   -> FPGA pin 25
//     ESP32 GPIO23 (MOSI) -> FPGA pin 28     ESP32 GPIO19 (MISO) <- FPGA pin 26
// Change PIN_* below if your board uses different pins.
//
// The key must equal KEY in voltquant_spi_top.v (public NIST test key for now).
// ============================================================================
#include <SPI.h>
#include <Preferences.h>
#include "voltquant_frame.h"

static const uint8_t KEY[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
};

#define PIN_SCK   18
#define PIN_MISO  19
#define PIN_MOSI  23
#define PIN_CS     5
#define SPI_HZ     1000000      // keep <= 2 MHz

static Preferences prefs;
static uint32_t boot_id = 0;     // changes on every ESP32 restart
static uint32_t counter = 0;     // frames since this boot
static uint8_t  last_txn[VQ_TXN_LEN];
static bool     have_last = false;

// seq must only ever go UP (the FPGA rejects anything not greater than the
// last accepted seq of that cell). boot_id lives in flash and is bumped once
// per restart, so seq keeps increasing even if the ESP32 is reset:
//     seq = (boot_id << 20) | counter      (1 M frames per boot)
static uint32_t next_seq() {
    counter++;
    return ((boot_id & 0xFFFu) << 20) | (counter & 0xFFFFFu);
}

static const char *class_name(int c) {
    return c == 0 ? "SAFE" : c == 1 ? "WARNING" : c == 2 ? "DEGRADED" : "?";
}

static const char *reject_name(int c) {
    switch (c) {
        case VQ_REJ_REPLAY:    return "REPLAY";
        case VQ_REJ_BAD_TAG:   return "BAD_TAG";
        case VQ_REJ_RESERVED:  return "BAD_RESERVED_BITS";
        case VQ_REJ_BAD_CLASS: return "ILLEGAL_CLASS";
        default:               return "NONE";
    }
}

// returns -1 if the text is not a class
static int parse_class(String s) {
    s.trim();
    s.toLowerCase();
    if (s == "safe" || s == "normal" || s == "0") return VQ_CLASS_SAFE;
    if (s == "warning" || s == "risk" || s == "1")               return VQ_CLASS_WARNING;
    if (s == "degraded" || s == "2")              return VQ_CLASS_DEGRADED;
    return -1;
}

static void print_hex(const char *label, const uint8_t *p, int n) {
    Serial.print(label);
    for (int i = 0; i < n; i++) {
        if (p[i] < 16) Serial.print('0');
        Serial.print(p[i], HEX);
    }
    Serial.println();
}

// one 34-byte SPI transaction; the FPGA's STATUS comes back in the last byte
static void send_txn(const uint8_t txn[VQ_TXN_LEN]) {
    uint8_t rx[VQ_TXN_LEN];
    SPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
    digitalWrite(PIN_CS, LOW);
    delayMicroseconds(5);
    SPI.transferBytes(txn, rx, VQ_TXN_LEN);
    delayMicroseconds(5);
    digitalWrite(PIN_CS, HIGH);
    SPI.endTransaction();

    uint8_t st = rx[VQ_TXN_LEN - 1];
    Serial.print("RX status=0x");
    if (st < 16) Serial.print('0');
    Serial.print(st, HEX);
    if (!VQ_ST_VALID(st)) {
        Serial.println(" NO_RESULT (FPGA gave no verdict: wiring/SPI problem?)");
        return;
    }
    if (VQ_ST_ACCEPTED(st)) {
        Serial.print(" ACCEPTED trusted_class=");
        Serial.println(class_name(VQ_ST_CLASS(st)));
    } else {
        Serial.print(" REJECTED reason=");
        Serial.print(reject_name(VQ_ST_CODE(st)));
        Serial.print(" trusted_class=");
        Serial.println(class_name(VQ_ST_CLASS(st)));
    }
}

static void handle_line(String line) {
    line.trim();
    if (line.length() == 0) return;

    String up = line;
    up.toUpperCase();

    if (up == "REPLAY") {
        if (!have_last) { Serial.println("ERR nothing to replay yet"); return; }
        Serial.println("TX replay of previous transaction");
        send_txn(last_txn);
        return;
    }

    bool tamper = false;
    if (up.startsWith("TAMPER ")) {
        tamper = true;
        line = line.substring(7);
        line.trim();
    }

    int c1 = line.indexOf(',');
    if (c1 < 0) { Serial.println("ERR format: <class>,<cell_id>[,<soh_percent>]"); return; }
    int c2 = line.indexOf(',', c1 + 1);

    int cls = parse_class(line.substring(0, c1));
    if (cls < 0) { Serial.println("ERR class must be safe|warning|degraded"); return; }

    String cell_s = (c2 < 0) ? line.substring(c1 + 1) : line.substring(c1 + 1, c2);
    cell_s.trim();
    int cell = cell_s.toInt();
    if (cell < 0 || cell > 255) { Serial.println("ERR cell_id must be 0-255"); return; }

    uint8_t soh = 0;
    float soh_pct = -1.0f;               // -1 = not provided
    if (c2 >= 0) {
        soh_pct = line.substring(c2 + 1).toFloat();
        if (soh_pct < 0.0f || soh_pct > 100.0f) { Serial.println("ERR soh must be 0-100 (percent)"); return; }
        soh = vq_soh_from_percent(soh_pct);
    }

    uint32_t seq = next_seq();
    uint32_t ts  = millis();

    uint8_t frame[16], tag[16], txn[VQ_TXN_LEN];
    vq_pack_frame(frame, seq, (uint8_t)cell, (uint8_t)cls, soh, ts);
    if (vq_cmac16(KEY, frame, tag) != 0) { Serial.println("ERR crypto failure"); return; }
    vq_build_txn(txn, frame, tag);
    if (tamper) vq_tamper_txn(txn);

    if (!tamper) {                       // REPLAY re-sends the last GENUINE frame
        memcpy(last_txn, txn, VQ_TXN_LEN);
        have_last = true;
    }

    Serial.print("TX seq=");   Serial.print(seq);
    Serial.print(" cell=");    Serial.print(cell);
    Serial.print(" class=");   Serial.print(class_name(cls));
    Serial.print(" soh=");
    if (soh_pct < 0.0f) Serial.print("n/a"); else { Serial.print(soh_pct); Serial.print("%"); }
    Serial.println(tamper ? " (TAMPERED after tagging)" : "");
    print_hex("TX frame=", txn, 16);
    print_hex("TX tag  =", txn + 16, 16);
    send_txn(txn);
}

void setup() {
    Serial.begin(115200);
    pinMode(PIN_CS, OUTPUT);
    digitalWrite(PIN_CS, HIGH);
    SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);

    prefs.begin("vq", false);
    boot_id = prefs.getUInt("boot", 0) + 1;
    prefs.putUInt("boot", boot_id);
    prefs.end();

    Serial.println();
    Serial.print("VoltQuant ESP32 ready. boot_id=");
    Serial.println(boot_id);
    Serial.println("Send: <safe|warning|degraded>,<cell_id>[,<soh_percent>]   |   TAMPER <line>   |   REPLAY");
}

void loop() {
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        handle_line(line);
    }
}
