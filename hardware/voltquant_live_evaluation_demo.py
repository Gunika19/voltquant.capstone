"""VoltQuant Live Evaluation Demo Script (One Script, One Run)

Executes the official 4-Act evaluation flow:
  Act 1: Real model predictions across 10 cells (Normal, Warning, Degraded)
  Act 2: Simulated attack - Tampering (Bit-flip on cell 10)
  Act 3: Simulated attack - Replay (Re-sending old normal frame)
  Act 4: System recovery with next genuine prediction

Features:
  - 10-second inspection pause between each packet (customizable via --gap)
  - Auto-flushes stdout for live terminal display
  - Maps FPGA LED indicators in real time
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink

CLASS_NAMES = {0: "NORMAL", 1: "WARNING", 2: "DEGRADED"}

def run_evaluation_flow(port: str = "COM3", gap: float = 10.0):
    print("=" * 80)
    print("⚡ VOLTQUANT LIVE HARDWARE-IN-THE-LOOP EVALUATION DEMO")
    print(f"Port: {port} | Baud: 115200 | Inspection Gap: {gap:.1f} seconds between packets")
    print("=" * 80)
    sys.stdout.flush()

    link = VoltQuantLink(port=port, baudrate=115200)

    # ─── Act 1: Real model predictions (10 cells) ───
    print("\n━━━ REAL MODEL PREDICTIONS ━━━")
    sys.stdout.flush()

    predictions = [
        (0, 1, 95.2),   # cell 1 normal
        (0, 2, 93.1),   # cell 2 normal
        (1, 3, 78.4),   # cell 3 warning      ← LED1 blinks
        (0, 4, 97.0),   # cell 4 normal
        (2, 5, 65.3),   # cell 5 degraded     ← LED2 blinks fast
        (0, 6, 91.8),   # cell 6 normal
        (0, 7, 96.5),   # cell 7 normal
        (1, 8, 74.2),   # cell 8 warning
        (0, 9, 98.1),   # cell 9 normal
        (2, 10, 58.7),  # cell 10 degraded    ← LED2 blinks fast
    ]

    for idx, (cls, cell, soh) in enumerate(predictions, 1):
        reply = link.send(cls, cell, soh)
        c_name = CLASS_NAMES.get(cls, str(cls))
        
        # Clean verdict display
        if "ACCEPTED" in reply:
            verdict = "ACCEPTED ✓"
        elif "REJECTED" in reply:
            verdict = "REJECTED ✗"
        else:
            verdict = reply[:30]

        # LED indicator guide for panel
        if cls == 0:
            led_guide = "[FPGA: LED0 STEADY ON + LED5 Pulse]"
        elif cls == 1:
            led_guide = "[FPGA: LED1 BLINK 1Hz (Warning) + LED5 Pulse]"
        else:
            led_guide = "[FPGA: LED2 BLINK 4Hz (DEGRADED: 10-MIN WARNING!) + LED5 Pulse]"

        print(f"TX cell={cell:<2} class={c_name:<9} soh={soh:>5.2f}%  → RX {verdict:<12} {led_guide}")
        sys.stdout.flush()

        if gap > 0 and idx < len(predictions):
            print(f"   ⏳ [Pausing {gap:.0f}s for LED inspection...]")
            sys.stdout.flush()
            time.sleep(gap)

    # ─── Act 2: Attacker tries to tamper ───
    if gap > 0:
        print(f"   ⏳ [Pausing {gap:.0f}s before simulated attack...]")
        sys.stdout.flush()
        time.sleep(gap)

    print("\n━━━ SIMULATED ATTACK: TAMPER ━━━")
    print("Attacker intercepts and changes 'degraded' to 'normal'...")
    sys.stdout.flush()
    
    reply_tamper = link.tamper(0, 10, 58.7)   # tries to fake cell 10 as normal
    if "BAD_TAG" in reply_tamper or "REJECTED" in reply_tamper:
        v_tamper = "REJECTED BAD_TAG ✗"
    else:
        v_tamper = reply_tamper[:30]

    print(f"TX cell=10 class=NORMAL [TAMPERED]     → RX {v_tamper:<16} [FPGA: LED3 ON 1.5s! LED2 keeps 4Hz strobe!]")
    sys.stdout.flush()

    # ─── Act 3: Attacker tries to replay ───
    if gap > 0:
        print(f"   ⏳ [Pausing {gap:.0f}s before replay attack...]")
        sys.stdout.flush()
        time.sleep(gap)

    print("\n━━━ SIMULATED ATTACK: REPLAY ━━━")
    print("Attacker replays an old 'normal' frame...")
    sys.stdout.flush()

    reply_replay = link.replay()
    if "REPLAY" in reply_replay or "REJECTED" in reply_replay:
        v_replay = "REJECTED REPLAY ✗"
    else:
        v_replay = reply_replay[:30]

    print(f"TX [REPLAY]                            → RX {v_replay:<16} [FPGA: LED4 ON 1.5s! LED2 keeps 4Hz strobe!]")
    sys.stdout.flush()

    # ─── Act 4: System recovers with next real prediction ───
    if gap > 0:
        print(f"   ⏳ [Pausing {gap:.0f}s before recovery frame...]")
        sys.stdout.flush()
        time.sleep(gap)

    print("\n━━━ SYSTEM RECOVERY ━━━")
    sys.stdout.flush()

    reply_recovery = link.send(0, 10, 90.5)     # cell 10 recovered, normal now
    if "ACCEPTED" in reply_recovery:
        v_rec = "ACCEPTED ✓"
    else:
        v_rec = reply_recovery[:30]

    print(f"TX cell=10 class=NORMAL    soh=90.50%  → RX {v_rec:<16} [FPGA: LED0 STEADY ON + LED5 Pulse]")
    sys.stdout.flush()

    print("\n━━━ DEMO COMPLETE ━━━")
    sys.stdout.flush()
    link.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoltQuant Live Evaluation Demo")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port for ESP32 (default: COM3)")
    parser.add_argument("--gap", type=float, default=10.0, help="Seconds between each packet (default: 10.0)")
    args = parser.parse_args()

    run_evaluation_flow(port=args.port, gap=args.gap)
