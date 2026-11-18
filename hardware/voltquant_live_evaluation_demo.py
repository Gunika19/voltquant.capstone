"""VoltQuant: Master Live Evaluation Script (hardware/voltquant_live_evaluation_demo.py)

Follows the definitive 4-Act evaluation flow for the capstone panel:
  Software Laptop (Model Predictions)
          │
          │  USB Serial (115200 baud, COM3)
          ▼
    ESP32 Dev Board  ─── SPI (1 MHz) ───  Tang Nano 20K FPGA (LEDs)

Acts:
  Act 1: Real Model Predictions (10 battery cells) -> LED0 (Normal), LED1 (Warning), LED2 (Degraded)
  Act 2: Simulated Attack: Tamper (Bit-flip on cell 10) -> LED3 Solid ON (BAD_TAG Rejected)
  Act 3: Simulated Attack: Replay (Stale frame replayed) -> LED4 Solid ON (REPLAY Rejected)
  Act 4: System Recovery (Fresh prediction arrives)     -> LED0 Steady ON (Recovered)
"""

from __future__ import annotations
import sys
import time
from datetime import datetime
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink

# Visual styling
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

CLASS_NAMES = {0: "NORMAL", 1: "WARNING", 2: "DEGRADED"}


def get_ts() -> str:
    """Return high-precision millisecond timestamp string [HH:MM:SS.mmm]."""
    return datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "]"


def parse_verdict_badge(reply: str) -> str:
    """Format verdict into clean, high-impact terminal string."""
    if "ACCEPTED" in reply:
        return f"{GREEN}{BOLD}→ RX ACCEPTED ✓{RESET}"
    elif "BAD_TAG" in reply:
        return f"{RED}{BOLD}→ RX REJECTED BAD_TAG ✗{RESET}"
    elif "REPLAY" in reply:
        return f"{RED}{BOLD}→ RX REJECTED REPLAY ✗{RESET}"
    elif "REJECTED" in reply or "ERR" in reply:
        return f"{RED}{BOLD}→ RX REJECTED ✗{RESET}"
    else:
        return f"{YELLOW}→ RX {reply[:25]}{RESET}"


def run_live_evaluation(port: str = "COM3", step_delay: float = 1.0):
    start_ts = get_ts()
    print("=" * 95)
    print(f"{CYAN}{BOLD}⚡ VOLTQUANT CAPSTONE HARDWARE EVALUATION DEMO{RESET}")
    print(f"Target Port : {port} (Baud: 115200) | Session Start: {start_ts}")
    print(f"Active Script: hardware/voltquant_live_evaluation_demo.py")
    print("=" * 95)

    link = VoltQuantLink(port=port, baudrate=115200)

    # ─── Act 1: Real model predictions (10 cells) ───
    print(f"\n{BOLD}━━━ REAL MODEL PREDICTIONS ━━━{RESET}")
    predictions = [
        (0, 1, 95.2),   # cell 1 normal
        (0, 2, 93.1),   # cell 2 normal
        (1, 3, 78.4),   # cell 3 warning      ← LED1 blinks 1 Hz
        (0, 4, 97.0),   # cell 4 normal
        (2, 5, 65.3),   # cell 5 degraded     ← LED2 blinks fast 4 Hz (10-min warning!)
        (0, 6, 91.8),   # cell 6 normal
        (0, 7, 96.5),   # cell 7 normal
        (1, 8, 74.2),   # cell 8 warning      ← LED1 blinks 1 Hz
        (0, 9, 98.1),   # cell 9 normal
        (2, 10, 58.7),  # cell 10 degraded    ← LED2 blinks fast 4 Hz (deep rollover)
    ]

    for cls, cell, soh in predictions:
        c_name = CLASS_NAMES.get(cls, str(cls))
        t_before = time.perf_counter()
        reply = link.send(cls, cell, soh)
        latency_ms = (time.perf_counter() - t_before) * 1000.0
        badge = parse_verdict_badge(reply)
        ts = get_ts()
        print(f"{ts} TX cell={cell:<2}  class={c_name:<8}  soh={soh:>5.2f}%  {badge}  ({latency_ms:>4.1f} ms)")
        time.sleep(step_delay)  # pause so panel can see LEDs change on FPGA

    # ─── Act 2: Attacker tries to tamper ───
    print(f"\n{BOLD}━━━ SIMULATED ATTACK: TAMPER ━━━{RESET}")
    print(f"{get_ts()} Attacker intercepts and changes 'degraded' to 'normal'...")
    t_before = time.perf_counter()
    reply_tamper = link.tamper(0, 10, 58.7)   # tries to fake cell 10 as normal
    lat_tamper = (time.perf_counter() - t_before) * 1000.0
    badge_t = parse_verdict_badge(reply_tamper)
    print(f"{get_ts()} TX cell=10 class=NORMAL [TAMPERED]     {badge_t}  ({lat_tamper:>4.1f} ms)")
    print(f"   {YELLOW}↳ Panel sees: LED3 turns ON for 1.5s; LED2 keeps blinking (still degraded!){RESET}")
    time.sleep(2.0)

    # ─── Act 3: Attacker tries to replay ───
    print(f"\n{BOLD}━━━ SIMULATED ATTACK: REPLAY ━━━{RESET}")
    print(f"{get_ts()} Attacker replays an old 'normal' frame...")
    t_before = time.perf_counter()
    reply_replay = link.replay()
    lat_replay = (time.perf_counter() - t_before) * 1000.0
    badge_r = parse_verdict_badge(reply_replay)
    print(f"{get_ts()} TX [REPLAY]                            {badge_r}  ({lat_replay:>4.1f} ms)")
    print(f"   {YELLOW}↳ Panel sees: LED4 turns ON for 1.5s; LED2 keeps blinking (still degraded!){RESET}")
    time.sleep(2.0)

    # ─── Act 4: System recovers with next real prediction ───
    print(f"\n{BOLD}━━━ SYSTEM RECOVERY ━━━{RESET}")
    t_before = time.perf_counter()
    reply_recover = link.send(0, 10, 90.5)     # cell 10 recovered, normal now
    lat_rec = (time.perf_counter() - t_before) * 1000.0
    badge_rec = parse_verdict_badge(reply_recover)
    print(f"{get_ts()} TX cell=10 class=NORMAL    soh=90.50%  {badge_rec}  ({lat_rec:>4.1f} ms)")
    print(f"   {GREEN}↳ Panel sees: LED0 steady ON (Battery normal state restored){RESET}")

    end_ts = get_ts()
    print(f"\n{BOLD}{CYAN}━━━ DEMO COMPLETE at {end_ts} ━━━{RESET}\n")
    link.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoltQuant Live Evaluation Demo")
    parser.add_argument("--port", type=str, default="COM3", help="Serial COM port (default: COM3)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to pause between cells (default: 1.0s)")
    args = parser.parse_args()

    run_live_evaluation(port=args.port, step_delay=args.delay)
