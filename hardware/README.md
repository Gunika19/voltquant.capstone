# VoltQuant Hardware-in-the-Loop (HIL) Integration Guide

This directory contains the firmware, communication protocols, and Python interface scripts connecting VoltQuant's machine learning prognostics pipeline to the **ESP32 Gateway Microcontroller** and **FPGA Hardware Verification Engine**.

---

## 1. System Architecture

```
+------------------------------------+
|        Host PC (VoltQuant)         |
|  - PG-HQML / SOH Prognostics       |
|  - vq_send.py (VoltQuantLink)      |
+------------------------------------+
                  |
                  | USB-UART Serial (115200 Baud, e.g. COM5)
                  v
+------------------------------------+
|     ESP32 Gateway Controller       |
|  - Firmware: voltquant_bridge.ino  |
|  - Packet parsing & bounds check   |
+------------------------------------+
                  |
                  | Hardware UART2 / SPI (TX=17, RX=16)
                  v
+------------------------------------+
|  FPGA Hardware Verification Engine |
|  - Cryptographic / telemetry check |
|  - Returns: ACCEPTED / REJECTED    |
+------------------------------------+
```

---

## 2. Packet Specification & Communication Protocol

### Inbound Telemetry Packet (PC $\to$ ESP32)
Each cycle prediction is transmitted as a single ASCII line terminated with `\n`:
```text
<class>,<cell_id>,<value>\n
```

* **`<class>` (Integer):** Alert state classification:
  * `0`: Normal Operation ($SOH \ge 80\%$)
  * `1`: Advisory / Warning ($70\% \le SOH < 80\%$)
  * `2`: Critical Egress / Rollover Precursor ($SOH < 70\%$)
* **`<cell_id>` (Integer):** Integer battery cell identifier:
  * e.g., `5` for Cell B0005 (or numeric index 1 to 124).
* **`<value>` (Integer):** State of Health (SOH) percentage:
  * e.g., `58` for 58% SOH (values range from 0 to 100).

**Example Packet:**
```text
1,5,58\n
```

### Outbound Verification Response (ESP32/FPGA $\to$ PC)
* **`ACCEPTED`**: The FPGA successfully validated the telemetry packet, confirmed physical boundary constraints, and verified cryptographic hash integrity.
* **`REJECTED`**: Telemetry failed validation (e.g., out-of-physical-bounds SOH, non-monotonic anomaly, or checksum mismatch).
* **`NO_RESULT`**: Hardware timeout (FPGA did not acknowledge within 100 ms).
* **`ERR`**: UART framing error or disconnected COM port.

---

## 3. Quick Start & Python Usage

### Sending Predictions from Python
```python
from vq_send import VoltQuantLink

# Connect to physical ESP32 on COM5 (auto-falls back to simulation if unplugged)
link = VoltQuantLink(port="COM5", baudrate=115200)

# Transmit prediction: Class 1 (Warning), Cell 5, SOH 58%
reply = link.send(cls=1, cell_id=5, value=58)

if "ACCEPTED" in reply:
    print("FPGA verified telemetry successfully:", reply)
elif "REJECTED" in reply or "NO_RESULT" in reply or "ERR" in reply:
    print("Hardware fault detected:", reply)

link.close()
```

---

## 4. Hardware Setup & Flashing Instructions

### Pinout Connections
| ESP32 Pin | FPGA Pin | Description |
| :--- | :--- | :--- |
| **GPIO 17 (TX2)** | RX Pin | ESP32 transmits formatted packet to FPGA |
| **GPIO 16 (RX2)** | TX Pin | ESP32 receives verification reply from FPGA |
| **GND** | GND | Common ground reference (**Mandatory**) |
| **GPIO 2** | Built-in LED | Blinks Blue on `ACCEPTED` packet |
| **GPIO 4** | External Red LED | Blinks Red on `REJECTED` or fault |

### Flashing the ESP32 Firmware
1. Open the Arduino IDE or PlatformIO.
2. Select Board: **ESP32 Dev Module**.
3. Open `hardware/esp32_firmware/voltquant_esp32_bridge.ino`.
4. Connect your ESP32 board via USB.
5. Click **Upload**.

---

## 5. Running the Hardware-in-the-Loop (HIL) Demonstration

To stream simulated or real battery discharge cycles through the hardware link:

```powershell
# Run with auto-detection on COM5 (or simulated if not connected)
python hardware/voltquant_hil_bridge.py --port COM5

# Run in fast mode
python hardware/voltquant_hil_bridge.py --port COM5 --fast
```
