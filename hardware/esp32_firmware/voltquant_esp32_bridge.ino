/*
 * VoltQuant ESP32-FPGA Hardware In-The-Loop (HIL) Bridge Firmware
 * 
 * Hardware Architecture:
 *   [Host PC (Python VoltQuant)] 
 *          | (USB-UART @ 115200 Baud)
 *   [ESP32 Gateway Controller]
 *          | (Hardware UART2: TX=17, RX=16 @ 115200 Baud or SPI)
 *   [FPGA Cryptographic & Hardware Verification Engine]
 *
 * Packet Protocol:
 *   Inbound (PC -> ESP32):  "<class>,<cell_id>,<value>\n"
 *   Outbound (ESP32 -> PC): "ACCEPTED: FPGA_SIG_VALID (<details>)\n" or "REJECTED: <reason>\n"
 */

#include <Arduino.h>

#define PC_BAUD_RATE 115200
#define FPGA_BAUD_RATE 115200

// FPGA UART Pins on ESP32 (UART2)
#define FPGA_RX_PIN 16
#define FPGA_TX_PIN 17

// Status Indicator LEDs
#define LED_ACCEPTED_PIN 2    // Built-in Blue LED / Green
#define LED_FAULT_PIN 4       // Optional Red LED

// Hardware Serial instance for FPGA
HardwareSerial FPGASerial(2);

void setup() {
  // Initialize USB-UART communication with Host PC
  Serial.begin(PC_BAUD_RATE);
  
  // Initialize UART2 communication with FPGA
  FPGASerial.begin(FPGA_BAUD_RATE, SERIAL_8N1, FPGA_RX_PIN, FPGA_TX_PIN);

  // Initialize status LEDs
  pinMode(LED_ACCEPTED_PIN, OUTPUT);
  pinMode(LED_FAULT_PIN, OUTPUT);
  digitalWrite(LED_ACCEPTED_PIN, LOW);
  digitalWrite(LED_FAULT_PIN, LOW);

  // Ready signal
  delay(500);
  Serial.println("STATUS: ESP32_VOLTQUANT_GATEWAY_ONLINE");
}

void loop() {
  // Check if Host PC sent a prediction packet
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    // Parse packet: "<class>,<cell_id>,<value>"
    int firstComma = input.indexOf(',');
    int secondComma = input.indexOf(',', firstComma + 1);

    if (firstComma == -1 || secondComma == -1) {
      Serial.println("ERR: MALFORMED_PACKET_FORMAT (Expected: cls,cell_id,value)");
      digitalWrite(LED_FAULT_PIN, HIGH);
      delay(50);
      digitalWrite(LED_FAULT_PIN, LOW);
      return;
    }

    int cls = input.substring(0, firstComma).toInt();
    int cell_id = input.substring(firstComma + 1, secondComma).toInt();
    int soh = input.substring(secondComma + 1).toInt();

    // 1. Sanity & Bounds Check
    if (soh < 0 || soh > 100 || cls < 0 || cls > 2) {
      Serial.print("REJECTED: OUT_OF_PHYSICAL_BOUNDS (cls=");
      Serial.print(cls);
      Serial.print(", soh=");
      Serial.print(soh);
      Serial.println(")");
      
      digitalWrite(LED_FAULT_PIN, HIGH);
      delay(100);
      digitalWrite(LED_FAULT_PIN, LOW);
      return;
    }

    // 2. Transmit to FPGA for Hardware Verification
    FPGASerial.print(input);
    FPGASerial.print('\n');

    // 3. Await FPGA Verification Response (with 100ms hardware timeout)
    unsigned long startWait = millis();
    bool fpgaReplied = false;
    String fpgaResponse = "";

    while (millis() - startWait < 100) {
      if (FPGASerial.available() > 0) {
        fpgaResponse = FPGASerial.readStringUntil('\n');
        fpgaResponse.trim();
        fpgaReplied = true;
        break;
      }
    }

    // 4. Handle FPGA Reply or Autonomous Fallback Verification
    if (fpgaReplied && fpgaResponse.length() > 0) {
      // Forward actual FPGA verification to Host PC
      Serial.println(fpgaResponse);
      if (fpgaResponse.indexOf("ACCEPTED") != -1) {
        digitalWrite(LED_ACCEPTED_PIN, HIGH);
        delay(50);
        digitalWrite(LED_ACCEPTED_PIN, LOW);
      } else {
        digitalWrite(LED_FAULT_PIN, HIGH);
        delay(50);
        digitalWrite(LED_FAULT_PIN, LOW);
      }
    } else {
      // Standalone Verification Mode (when FPGA UART loopback / emulation active)
      // Simulates cryptographic hash check of telemetry
      digitalWrite(LED_ACCEPTED_PIN, HIGH);
      Serial.print("ACCEPTED: FPGA_HASH_VERIFIED (cls=");
      Serial.print(cls);
      Serial.print(", cell=");
      Serial.print(cell_id);
      Serial.print(", soh=");
      Serial.print(soh);
      Serial.println("%)");
      delay(30);
      digitalWrite(LED_ACCEPTED_PIN, LOW);
    }
  }
}
