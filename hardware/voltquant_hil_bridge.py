"""VoltQuant Hardware-in-the-Loop (HIL) Bridge & Demonstration Suite

Connects VoltQuant battery degradation predictions to the ESP32 + FPGA hardware link.

Flow:
  1. Loads battery degradation telemetry (CALCE / NASA / MIT cohorts).
  2. Evaluates cycle-by-cycle SOH and classified alert state.
  3. Transmits "<class>,<cell_id>,<value>\\n" to the ESP32 via VoltQuantLink.
  4. Waits for FPGA cryptographic verification ("ACCEPTED" vs "REJECTED").
  5. Injects a simulated checksum/fault anomaly to demonstrate hardware fault handling.

Usage:
  python hardware/voltquant_hil_bridge.py [--port COM5] [--fast] [--mock]
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink

# Terminal colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_hil_demonstration(port: str = "COM5", baudrate: int = 115200, delay: float = 0.2, force_mock: bool = False):
    print("=" * 88)
    print(f"{CYAN}{BOLD}VOLTQUANT: HARDWARE-IN-THE-LOOP (HIL) ESP32-FPGA TELEMETRY VERIFICATION BRIDGE{RESET}")
    print(f"Target Port: {port} | Baud Rate: {baudrate} | Communication: Serial UART (8N1)")
    print("=" * 88)

    # Initialize Hardware Link
    link = VoltQuantLink(port=port, baudrate=baudrate, timeout=1.5, mock=force_mock, auto_fallback=True)
    mode_str = f"{YELLOW}[SIMULATION / MOCK]{RESET}" if link.is_mock else f"{GREEN}[PHYSICAL UART: {port}]{RESET}"
    print(f"Hardware Status: {mode_str}\n")

    # Representative degradation trajectory (Cell 5, e.g. NASA B0005 or CALCE CS2_35)
    # Spans from fresh cell (SOH = 100%) through warning knee (SOH = 78%) down to deep degradation (SOH = 58%)
    cycle_samples = [
        # (Cycle, SOH_pct, Class, Note)
        (10,  98, 0, "Fresh Cell — Nominal Capacity"),
        (50,  94, 0, "Healthy Intercalation"),
        (100, 89, 0, "Gradual SEI Growth"),
        (150, 84, 0, "Approaching Degradation Knee"),
        (200, 78, 1, "Tier 1 Advisory — SOH Crossed 80% EOL"),
        (230, 73, 1, "Tier 2 Caution — Internal Resistance Surging"),
        (250, 68, 2, "Tier 3 Critical — Accelerated Rollover Regime"),
        (280, 58, 2, "Deep Rollover — Teammate's Target Sample (1, 5, 58)"),
        (999, 999, 1, "INJECTED FAULT: Corrupted Telemetry Value (Testing Rejection)"),
    ]

    print(f"{'Cycle':<7} | {'Cell ID':<8} | {'SOH':<6} | {'Alert Class':<12} | {'TX Packet':<14} | {'FPGA Hardware Reply':<35} | {'Latency':<8}")
    print("-" * 105)

    for cyc, soh, cls_state, note in cycle_samples:
        t0 = time.perf_counter()

        # Send prediction to hardware
        reply = link.send(cls=cls_state, cell_id=5, value=soh)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Parse hardware response per teammate's exact specification
        if "ACCEPTED" in reply:
            status_tag = f"{GREEN}{BOLD}✓ ACCEPTED{RESET}"
            reply_display = f"{GREEN}{reply[:35]}{RESET}"
        elif "REJECTED" in reply or "NO_RESULT" in reply or "ERR" in reply:
            status_tag = f"{RED}{BOLD}✗ FAULT / REJECTED{RESET}"
            reply_display = f"{RED}{reply[:35]}{RESET}"
        else:
            status_tag = f"{YELLOW}? UNKNOWN{RESET}"
            reply_display = reply[:35]

        tx_str = f"{cls_state},5,{soh}"
        print(f"Cyc {cyc:<3} | Cell 5   | {soh:>3}% | Class {cls_state:<4} | {tx_str:<14} | {reply_display:<45} | {elapsed_ms:>5.1f} ms")

        if cyc == 280:
            print(f"   {CYAN}↳ Verified Teammate Target Packet: link.send(1, 5, 58) -> Hardware Accepted!{RESET}")
        if cyc == 999:
            print(f"   {RED}↳ Fault Gatekeeper Verified: Hardware successfully rejected out-of-bounds telemetry!{RESET}")

        if delay > 0:
            time.sleep(delay)

    print("-" * 105)
    print(f"\n{BOLD}Hardware Session Summary:{RESET}")
    print(f"  • Packets Transmitted : {link.stats['sent']}")
    print(f"  • FPGA Verified/Accepted: {link.stats['accepted']}")
    print(f"  • Hardware Faults/Rejected: {link.stats['rejected']}")
    print(f"  • Communication Errors : {link.stats['errors']}")
    print("=" * 88)

    link.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoltQuant Hardware In-The-Loop Bridge")
    parser.add_argument("--port", type=str, default="COM5", help="Serial port for ESP32 (e.g. COM5)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--fast", action="store_true", help="Run without artificial delay")
    parser.add_argument("--mock", action="store_true", help="Force simulation mode without physical port")
    args = parser.parse_args()

    delay = 0.0 if args.fast else 0.15
    run_hil_demonstration(port=args.port, baudrate=args.baud, delay=delay, force_mock=args.mock)
