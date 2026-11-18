# VoltQuant — Hardware/Software Integration Manual
**Bridge Guide for Software & Hardware Engineering Teams**

---

## 1. Executive Overview: Bridging the Gap Between HW and SW

The single most common confusion in the team is: *"Is the ML model running on the FPGA or on the laptop?"*

Here is the definitive architectural split:

| Layer | Component | Hardware Platform | What It Actually Does |
| :--- | :--- | :--- | :--- |
| **Software Layer** | **Prognostics Engine** | **Laptop** (PyTorch / PennyLane) | • Reconstructs battery discharge curves ($\Delta Q$ variance, resistance)<br>• Computes State of Health (SOH %)<br>• Classifies Alert State (`SAFE`, `WARNING`, `DEGRADED`) |
| **Bridge Layer** | **Gateway & Tag Generator** | **ESP32 Dev Module** | • Converts simple ASCII predictions into 128-bit frames<br>• Maintains monotonic sequence numbers in NVS flash (Anti-Replay)<br>• Computes 128-bit AES-CMAC tag with NIST key<br>• Transmits 34-byte transaction over SPI (Mode 0, 1 MHz) |
| **Hardware Layer** | **Trust Anchor & Verifier** | **Tang Nano 20K FPGA** (Gowin GW2AR-18C) | • **Tamper-evident alert verifier** (Does NOT run ML!)<br>• Re-computes AES-CMAC in hardware (11 clock cycles @ 27 MHz)<br>• Enforces per-cell sequence monotonicity (Replay Protection)<br>• Checks structure and reserved bit masks<br>• Drives physical status LEDs & returns status byte (`0xC0`, `0xC2`, `0xC4`) |

### Why This Architecture Wins in an Automotive Viva
If an evaluator asks: *"Why didn't you put the ML model inside the FPGA?"*
* **The Correct Answer:**
  > *"In an electric vehicle, safety-critical architecture strictly separates the **perception/inference engine** from the **safety-critical actuator/actuation gatekeeper**. Running complex ML or quantum circuits directly inside the FPGA would introduce massive logic complexity and an unverified attack surface. Instead, our FPGA acts as a **fixed-function cryptographic trust anchor**: it refuses to activate the vehicle's safety state machine unless the alert can be mathematically proven to be authentic, fresh, and untampered."*

---

## 2. End-to-End Data Flow: From Model Prediction to Physical LED

When the model runs cycle inference on battery cell 6 and detects $SOH = 71\%$ (Degraded):

```
[ Laptop: Python ML Model ]
      │
      │ 1. Model decides: cell=6, class=DEGRADED, SOH=71.0%
      │    Calls: link.send("DEGRADED", 6, 71.0)
      ▼
[ USB Serial: 115200 Baud ]
      │
      │ 2. Sends 15 ASCII bytes: "degraded,6,71\n"
      ▼
[ ESP32 Gateway: voltquant_esp32.ino ]
      │
      │ 3. Parses: class=2, cell_id=6, soh=71.0
      │ 4. Computes seq from NVS: seq = (boot_cnt << 20) | frame_cnt  (e.g., 0x00200004)
      │ 5. Bit-packs 16-byte (128-bit) frame:
      │    [seq (32b) | cell (8b) | class (2b) | SOH (8b) | timestamp (32b) | reserved (46b)]
      │ 6. Computes 16-byte AES-128-CMAC tag over the frame
      │ 7. Assembles 34-byte SPI transaction buffer: [16B frame | 16B tag | 2B padding]
      ▼
[ Hardware SPI: Mode 0, 1 MHz (SCK=18, MOSI=23, MISO=19, CS=5) ]
      │
      │ 8. Clocks 34 bytes out in 272 microseconds
      ▼
[ Tang Nano 20K FPGA: voltquant_spi_top.v ]
      │
      │ 9. spi_slave deserialises frame and tag
      │ 10. Parallel Pipeline (~0.44 microseconds):
      │     • frame_unpacker: Verifies reserved bits == 0 and class != 3
      │     • aes128_cmac: Re-computes AES tag and compares (tag_ok = 1)
      │     • seq_counter_guard: Checks seq > last_accepted_seq[6] (is_fresh = 1)
      │ 11. alert_fsm: Combines checks -> ACCEPT!
      │ 12. Sets status byte = 0xC4 (Accepted, Class Degraded)
      │ 13. Clocks status byte out on MISO during 34th byte
      ▼
[ Physical Tang Nano 20K LEDs ]
      • LED2 (Pin 17) -> RAPID 4 Hz STROBE (10-minute warning active!)
      • LED5 (Pin 20) -> Pulses for 1.0 second (Frame Accepted)
      ▼
[ ESP32 reads 0xC4 on MISO ]
      • Prints: "RX status=0xC4 ACCEPTED trusted_class=DEGRADED"
      ▼
[ Laptop vq_send.py ]
      • Returns reply string to Python model script
```

---

## 3. The Software API (`vq_send.py`): How to Call Hardware in Python

The software team does **not** need to worry about SPI timing, bit shifts, or cryptographic subkeys. Everything is encapsulated in `VoltQuantLink`:

### Basic Usage
```python
from vq_send import VoltQuantLink

# 1. Connect to ESP32 on COM3 (auto-falls back to simulation if unplugged)
link = VoltQuantLink(port="COM3", baudrate=115200)

# 2. Transmit normal prediction (Cycle 50, Cell 6, SOH 95%)
reply = link.send("SAFE", 6, 95.0)
print(reply)
# -> RX status=0xC0 ACCEPTED trusted_class=SAFE

# 3. Transmit critical rollover prediction (Cycle 300, Cell 6, SOH 58%)
reply = link.send("DEGRADED", 6, 58.0)
print(reply)
# -> RX status=0xC4 ACCEPTED trusted_class=DEGRADED
# -> FPGA LED2 instantly strobes at 4 Hz!

link.close()
```

### Live Attack Demonstrations for the Panel
```python
# Attack 1: Malicious Bit-Flip (Tampering)
# ESP32 computes tag, flips 1 bit in payload, transmits to FPGA:
tamper_reply = link.tamper("SAFE", 6, 95.0)
print(tamper_reply)
# -> RX status=0x94 REJECTED reason=BAD_TAG trusted_class=DEGRADED
# -> FPGA LED3 turns SOLID ON for 1.5s!

# Attack 2: Replay Attack (Stale Telemetry)
# Re-sends previous frame without incrementing sequence counter:
replay_reply = link.replay()
print(replay_reply)
# -> RX status=0x8C REJECTED reason=REPLAY trusted_class=DEGRADED
# -> FPGA LED4 turns SOLID ON for 1.5s!
```

---

## 4. Tang Nano 20K Physical LED Annunciator Map

The Tang Nano 20K has 6 on-board surface-mount LEDs (`LED0` through `LED5`) located along the edge of the board:

| LED | Physical Pin | Logic Condition | Visual Appearance | Meaning in Battery Context |
| :--- | :--- | :--- | :--- | :--- |
| **LED0** | **Pin 15** | `trusted_class == SAFE` | **Steady ON** | **Nominal State**: SOH $\ge 80\%$. Battery healthy. |
| **LED1** | **Pin 16** | `trusted_class == WARNING` | **Blinks at 1 Hz** (Slow pulse) | **Advisory Knee**: $70\% \le SOH < 80\%$. Internal resistance surge detected. |
| **LED2** | **Pin 17** | `trusted_class == DEGRADED` | **Strobes at 4 Hz** (Rapid flash) | **Critical Rollover Active**: $SOH < 70\%$. **10-minute advance warning buffer active before 72s voltage collapse!** |
| **LED3** | **Pin 18** | `reject_code == BAD_TAG` | **Solid ON for 1.5s** | **Tamper Attack Blocked**: FPGA cryptographic tag validation failed. |
| **LED4** | **Pin 19** | `reject_code == REPLAY` | **Solid ON for 1.5s** | **Replay Attack Blocked**: Duplicate/stale sequence number rejected. |
| **LED5** | **Pin 20** | `accepted == 1` | **Flashes for 1.0s** | **Heartbeat ACK**: Pulses on every valid frame acknowledged by FPGA. |

---

## 5. Wiring Schematic (Only 5 Jumper Wires Needed!)

```
  ESP32 Dev Board (30-Pin)                Tang Nano 20K FPGA
+--------------------------+            +--------------------+
|  GPIO 18 (SCK)           | ---------> |  Pin 27 (spi_sclk) |
|  GPIO 23 (MOSI)          | ---------> |  Pin 28 (spi_mosi) |
|  GPIO 19 (MISO)          | <--------- |  Pin 26 (spi_miso) |
|  GPIO 5  (CS)            | ---------> |  Pin 25 (spi_cs_n) |
|  GND                     | <--------> |  GND               |
+--------------------------+            +--------------------+
```
* **Power**: ESP32 is powered via Laptop USB. Tang Nano 20K is powered via its own USB-C cable (from laptop or phone charger).
* **Ground**: The common GND wire is **mandatory** for logic reference.

---

## 6. How to Run the Demonstration in 2 Minutes

### Method 1: The Automated Model-to-Hardware Stream
Run the end-to-end degradation flight path with live visual pauses:
```powershell
python hardware/voltquant_model_to_hardware_stream.py --port COM3 --delay 2.0
```
This streams 8 cycles showing:
1. Nominal health (`LED0` steady ON).
2. SOH 80% advisory knee (`LED1` blinking at 1 Hz).
3. SOH 58% deep rollover (`LED2` rapid 4 Hz strobe — 10-minute warning active).
4. Automated `TAMPER` attack (`LED3` solid ON for 1.5s).
5. Automated `REPLAY` attack (`LED4` solid ON for 1.5s).

### Method 2: The Fast 3-Packet Verification Check
For a quick 15-second sanity check:
```powershell
python hardware/demo_3_packets.py COM3
```

---

## 7. Viva Q&A Cheat Sheet (For Evaluators & Mentors)

1. **"Why didn't you just encrypt the telemetry?"**
   * *Answer:* Encryption provides *confidentiality* (hiding data), not *integrity* or *authenticity*. An encrypted alert can still be modified by bit-flipping, producing decrypted garbage. We need tamper evidence, which is why AES-CMAC is the industry standard (RFC 4493).
2. **"What prevents an attacker from replaying a recorded 'SAFE' packet tomorrow?"**
   * *Answer:* Every frame contains a 32-bit sequence counter. The FPGA maintains a per-cell table in internal SRAM tracking the highest sequence number ever accepted. If an attacker replays yesterday's frame, the sequence number is lower than or equal to the stored value, and the FPGA rejects it with `REJECTED reason=REPLAY` (`LED4`).
3. **"What happens if the ESP32 is rebooted?"**
   * *Answer:* The ESP32 maintains a persistent boot counter in NVS flash. On every reboot, the sequence number jumps forward by $2^{20}$ ($1,048,576$), guaranteeing that sequence numbers never roll backward across power cycles.
4. **"What is the physical latency of the hardware verification?"**
   * *Answer:* The 10-round AES-CMAC engine in Verilog completes in 11 clock cycles at 27 MHz ($\sim 0.44\ \mu\text{s}$). The full round-trip from Laptop USB to ESP32 SPI to FPGA and back takes **17.8 milliseconds**, well within the 100 ms real-time deadline for automotive battery management systems.
