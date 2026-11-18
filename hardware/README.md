# VoltQuant Hardware-in-the-Loop (HIL) Integration & Live Demo Guide

This directory contains the firmware, communication protocols, and Python interface scripts connecting VoltQuant's machine learning prognostics pipeline to the **ESP32 Gateway Microcontroller** and **FPGA Hardware Verification Engine**, driving the real-time **Automotive Cockpit Safety Dashboard**.

---

## 1. End-to-End System Architecture

```
+------------------------------------+
|             LAPTOP 1               |
|      (Battery ML Model Host)       |
|  - PG-HQML / SOH Prognostics       |
|  - vq_send.py (VoltQuantLink)      |
+------------------------------------+
                  |
                  | USB-UART Serial (115200 Baud, e.g. COM5)
                  v
+------------------------------------+
|     ESP32 Gateway Controller       |
|  - Firmware: voltquant_bridge.ino  |
|  - ASCII packet parsing & bounds   |
+------------------------------------+
                  |
                  | Hardware UART2 (TX=17, RX=16, 115200 Baud)
                  v
+------------------------------------+
|  FPGA Hardware Verification Engine |
|  - Cryptographic / telemetry check |
|  - Monotonicity bounds validation  |
|  - Returns: ACCEPTED / REJECTED    |
+------------------------------------+
                  |
                  | Serial UART or Local LAN / Wi-Fi Bridge
                  v
+------------------------------------+
|             LAPTOP 2               |
|    (Cockpit Safety Dashboard)      |
|  - Receiver: voltquant_receiver.py |
|  - Web UI: http://localhost:5000   |
|  - 10-Min Advance Warning Buffer   |
|  - SOH Gauge & FPGA Audit Badge    |
+------------------------------------+
```

---

## 2. Hardware File Locations in Repository

| File Path | Role | Description |
| :--- | :--- | :--- |
| [`vq_send.py`](../vq_send.py) | Root Driver | Core client interface implementing `VoltQuantLink(port)` for sending `<class>,<cell_id>,<value>` packets. |
| [`hardware/vq_send.py`](vq_send.py) | Symlink / Mirror | Hardware directory copy of `VoltQuantLink`. |
| [`hardware/esp32_firmware/voltquant_esp32_bridge.ino`](esp32_firmware/voltquant_esp32_bridge.ino) | Microcontroller Firmware | C++ firmware for ESP32 Dev Module; parses serial commands, forwards to FPGA over UART2, and toggles status LEDs. |
| [`hardware/voltquant_hil_bridge.py`](voltquant_hil_bridge.py) | **Laptop 1** Test Streamer | Automated test harness that replays battery cell degradation trajectories into the ESP32 via `VoltQuantLink`. |
| [`hardware/dashboard/voltquant_receiver_dashboard.py`](dashboard/voltquant_receiver_dashboard.py) | **Laptop 2** Cockpit Dashboard | High-contrast automotive driver display serving on `http://localhost:5000` with real-time SOH gauge, FPGA verification badge, and flashing 10-minute warning. |
| [`tests/test_hardware_link.py`](../tests/test_hardware_link.py) | CI Test Suite | Automated unit tests validating packet formatting, boundary checks, and error handling. |

---

## 3. Communication Protocol Specification

### Inbound Telemetry Packet (Laptop 1 $\to$ ESP32)
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

### Outbound Verification Response (ESP32/FPGA $\to$ Laptop 1)
* **`ACCEPTED`**: The FPGA successfully validated the telemetry packet, confirmed physical boundary constraints, and verified cryptographic hash integrity.
* **`REJECTED`**: Telemetry failed validation (e.g., out-of-physical-bounds SOH, non-monotonic anomaly, or checksum mismatch).
* **`NO_RESULT`**: Hardware timeout (FPGA did not acknowledge within 100 ms).
* **`ERR`**: UART framing error or disconnected COM port.

---

## 4. Hardware Wiring & Pinout Guide

### Wiring ESP32 to FPGA
| ESP32 Pin | FPGA Pin | Description |
| :--- | :--- | :--- |
| **GPIO 17 (TX2)** | RX Pin | ESP32 transmits formatted packet to FPGA |
| **GPIO 16 (RX2)** | TX Pin | ESP32 receives verification reply from FPGA |
| **GND** | GND | Common ground reference (**Mandatory**) |
| **GPIO 2** | Built-in LED | Blinks Blue on `ACCEPTED` packet |
| **GPIO 4** | External Red LED | Blinks Red on `REJECTED` or fault |

### Wiring Laptop 1 and Laptop 2
* **Laptop 1**: Connects to the **ESP32 Micro-USB / Type-C port** (enumerates as a COM port, e.g., `COM5` on Windows or `/dev/ttyUSB0` on Linux).
* **Laptop 2**: Connects to the **FPGA USB-UART port** (e.g., `COM3`), or receives telemetry over the local Wi-Fi / Hotspot network (`POST http://<LAPTOP_2_IP>:5000/api/telemetry`).

---

## 5. Step-by-Step Live Demonstration Guide (Two Laptops)

### Step A: Flash the ESP32
1. Open the Arduino IDE or PlatformIO.
2. Select Board: **ESP32 Dev Module**.
3. Open `hardware/esp32_firmware/voltquant_esp32_bridge.ino`.
4. Connect the ESP32 via USB and click **Upload**.
5. Once uploaded, connect ESP32 pins `TX2 (17)` and `RX2 (16)` to the FPGA.

---

### Step B: Setup Laptop 2 (Cockpit Dashboard & Warning Station)
1. On Laptop 2, clone the repository or copy the `hardware/` folder.
2. Launch the receiver dashboard:
   ```powershell
   # If connected to FPGA over USB-UART:
   python hardware/dashboard/voltquant_receiver_dashboard.py --port COM3 --web-port 5000

   # If running standalone or over Wi-Fi network:
   python hardware/dashboard/voltquant_receiver_dashboard.py --web-port 5000
   ```
3. Open a browser on Laptop 2 and navigate to:
   ```text
   http://localhost:5000
   ```
4. **Dashboard Features on Display:**
   * **Real-time SOH Gauge:** Smoothly decrements as the battery discharges.
   * **Multi-Tier Classification:** Displays `TIER 0: NORMAL`, `TIER 1/2: ADVISORY`, and `TIER 3: CRITICAL EGRESS`.
   * **10-Minute Advance Warning Buffer:** Flashes a high-visibility red emergency banner:
     $$\text{CRITICAL WARNING: 10.0 MIN ADVANCE EGRESS BUFFER (VOLTAGE COLLAPSE IMMINENT)}$$
   * **FPGA Audit Badge:** Shows `FPGA: ACCEPTED & AUTHENTICATED` in real time.

---

### Step C: Setup Laptop 1 (ML Model Transmitter)
1. On Laptop 1, open PowerShell / Terminal in the repository root.
2. Test sending a single prediction packet using Python:
   ```python
   from vq_send import VoltQuantLink

   # Connect to physical ESP32 on COM5
   link = VoltQuantLink(port="COM5", baudrate=115200)

   # Transmit prediction: Class 1 (Warning), Cell 5, SOH 58%
   reply = link.send(cls=1, cell_id=5, value=58)
   print("FPGA Reply:", reply)
   link.close()
   ```
3. Or run the full automated discharge stream:
   ```powershell
   # Stream degradation trajectory through ESP32 and FPGA to Laptop 2
   python hardware/voltquant_hil_bridge.py --port COM5
   ```
4. As predictions stream:
   * Laptop 1 receives `ACCEPTED` acknowledgments from the FPGA.
   * ESP32 blinks blue on valid packets.
   * Laptop 2 immediately updates its visual cockpit gauge and displays the 10-minute thermal/voltage runaway precursor alert!
