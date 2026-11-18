"""VoltQuant 3-Packet Physical LED Verification Demo

Streams 3 distinct packets to the ESP32 on COM3 with pauses between them
so you can visually inspect the Tang Nano 20K onboard LEDs:

Packet 1: SAFE (SAFE, 5, 95%)
  -> Tang Nano LED0: Steady ON (Normal / Safe state)
  -> Tang Nano LED5: Pulses for 1.0s (Frame Accepted)

Packet 2: DEGRADED (DEGRADED, 5, 58%)
  -> Tang Nano LED2: Rapid 4 Hz Strobe (Critical rollover precursor!)
  -> Tang Nano LED5: Pulses for 1.0s (Frame Accepted)

Packet 3: TAMPER (TAMPER SAFE, 5, 95%)
  -> Tang Nano LED3: Solid ON for 1.5s (Bad Tag Cryptographic Reject)
  -> Tang Nano LED5: OFF (Rejected, no accept pulse)
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink

def run_3_packet_demo(port="COM3"):
    print("=" * 80)
    print("⚡ VOLTQUANT PHYSICAL LED VERIFICATION DEMO (TANG NANO 20K FPGA)")
    print(f"Connecting to ESP32 on {port}...")
    print("=" * 80)

    link = VoltQuantLink(port=port, baudrate=115200)

    # -------------------------------------------------------------------------
    # STEP 1: SAFE PACKET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("👉 [TEST 1/3] TRANSMITTING: SAFE PACKET ('SAFE,5,95.0')")
    print("   Sending genuine healthy battery telemetry (SOH = 95%)...")
    reply1 = link.send("SAFE", 5, 95.0)
    print(f"   Hardware Reply: {reply1}")
    print("\n   👀 LOOK AT YOUR TANG NANO 20K BOARD NOW:")
    print("   ✓ LED0 (Pin 15) -> STEADY ON (Normal / Safe battery health)")
    print("   ✓ LED5 (Pin 20) -> FLASHES for 1.0s (Acknowledging accepted packet)")
    print("=" * 80)
    print("   (Pausing 5 seconds so you can verify the LEDs...)")
    time.sleep(5.0)

    # -------------------------------------------------------------------------
    # STEP 2: DEGRADED PACKET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("👉 [TEST 2/3] TRANSMITTING: DEGRADED PACKET ('DEGRADED,5,58.0')")
    print("   Sending rollover precursor telemetry (SOH = 58%)...")
    reply2 = link.send("DEGRADED", 5, 58.0)
    print(f"   Hardware Reply: {reply2}")
    print("\n   👀 LOOK AT YOUR TANG NANO 20K BOARD NOW:")
    print("   ✓ LED2 (Pin 17) -> RAPID 4 Hz STROBE (Flashing fast!)")
    print("                      Signaling critical rollover / 10-minute advance warning buffer!")
    print("   ✓ LED5 (Pin 20) -> FLASHES for 1.0s (Acknowledging accepted packet)")
    print("=" * 80)
    print("   (Pausing 5 seconds so you can verify the LEDs...)")
    time.sleep(5.0)

    # -------------------------------------------------------------------------
    # STEP 3: TAMPERED PACKET (BIT-FLIP ATTACK)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("👉 [TEST 3/3] TRANSMITTING: TAMPERED PACKET ('TAMPER SAFE,5,95.0')")
    print("   ESP32 generates valid AES-CMAC tag, then deliberately flips 1 payload bit...")
    reply3 = link.tamper("SAFE", 5, 95.0)
    print(f"   Hardware Reply: {reply3}")
    print("\n   👀 LOOK AT YOUR TANG NANO 20K BOARD NOW:")
    print("   ✓ LED3 (Pin 18) -> SOLID ON for 1.5 seconds!")
    print("                      Hardware AES-CMAC validator caught the bad tag!")
    print("   ✗ LED5 (Pin 20) -> OFF (Packet correctly rejected)")
    print("=" * 80)

    link.close()
    print("\n✅ 3-Packet LED Verification Complete!")

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    run_3_packet_demo(port)
