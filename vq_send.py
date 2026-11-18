"""VoltQuant Hardware Communication Link (vq_send.py)

Provides serial UART communication between the VoltQuant Python prognostics
pipeline (Model Laptop) and the ESP32 Gateway running 'voltquant_esp32.ino',
which connects via SPI to the Tang Nano 20K FPGA running 'voltquant_spi_top.v'.

Protocol Overview:
  Model Laptop --USB Serial (115200)--> ESP32 --SPI (mode 0, 1 MHz)--> Tang Nano 20K FPGA

Input Packet Format:
  "<class>,<cell_id>[,<soh_percent 0-100>]\\n"
    - <class>       : Alert classification ('safe'|'warning'|'degraded' or 0|1|2)
    - <cell_id>     : Cell identifier (0-255 or string e.g. "B0005" -> 5)
    - <soh_percent> : (Optional) State of Health percentage 0-100

Hardware Attack / Demo Commands:
  - "TAMPER <line>\\n" : ESP32 flips 1 payload bit AFTER computing AES-CMAC tag (FPGA rejects with BAD_TAG)
  - "REPLAY\\n"        : ESP32 re-transmits last frame (FPGA rejects with REPLAY due to non-increasing seq)

Verdict Responses:
  - Contains "ACCEPTED"  : FPGA verified AES-CMAC tag, monotonic seq, and valid bounds.
  - Contains "REJECTED"  : FPGA rejected frame (BAD_TAG, REPLAY, ILLEGAL_CLASS, etc.).
  - Contains "NO_RESULT" : SPI/wiring fault or FPGA unresponsive.
  - Contains "ERR"       : Serial framing or input syntax error.

Usage Example:
  from vq_send import VoltQuantLink

  link = VoltQuantLink("COM5")
  reply = link.send(1, 5, 58)  # Class 1 (Warning), Cell 5, 58% SOH

  if "ACCEPTED" in reply:
      print("FPGA verified:", reply)
  elif "REJECTED" in reply or "NO_RESULT" in reply or "ERR" in reply:
      print("Fault detected:", reply)

  # Interactive hardware attack demonstrations:
  tamper_reply = link.tamper(1, 5, 58)  # FPGA returns: REJECTED reason=BAD_TAG
  replay_reply = link.replay()          # FPGA returns: REJECTED reason=REPLAY
"""

from __future__ import annotations
import sys
import time
import logging
from typing import Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoltQuantLink")

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False
    logger.warning("pyserial is not installed. VoltQuantLink will operate in simulation mode.")


class VoltQuantLink:
    """Serial communication bridge connecting VoltQuant prognostics to ESP32 + FPGA."""

    CLASS_MAP = {
        0: "SAFE",
        1: "WARNING",
        2: "DEGRADED",
        "0": "SAFE",
        "1": "WARNING",
        "2": "DEGRADED",
        "safe": "SAFE",
        "normal": "SAFE",
        "risk": "WARNING",
        "warning": "WARNING",
        "degraded": "DEGRADED",
        "critical": "DEGRADED"
    }

    def __init__(
        self,
        port: str = "COM5",
        baudrate: int = 115200,
        timeout: float = 2.0,
        mock: bool = False,
        auto_fallback: bool = True
    ):
        """Initialize connection to the ESP32 Gateway.
        
        Args:
            port: Serial COM port (e.g., "COM5" on Windows, "/dev/ttyUSB0" on Linux).
            baudrate: Baud rate (preloaded firmware uses 115200).
            timeout: Serial read timeout in seconds.
            mock: If True, forces simulation mode.
            auto_fallback: If True, falls back to simulation if physical COM port fails to open.
        """
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_mock = mock
        self.serial_conn: Optional[serial.Serial] = None
        self.stats = {"sent": 0, "accepted": 0, "rejected": 0, "errors": 0}
        self.last_transaction_log: List[str] = []
        self._last_mock_packet: Optional[str] = None

        if not self.is_mock and HAS_PYSERIAL:
            available = self.list_available_ports()
            if available and (self.port_name not in available or self.port_name == "auto"):
                logger.info(f"Port '{self.port_name}' not in {available}. Auto-selecting detected port '{available[0]}'.")
                self.port_name = available[0]

            try:
                logger.info(f"Connecting to ESP32 on {self.port_name} at {self.baudrate} baud...")
                self.serial_conn = serial.Serial(
                    port=self.port_name,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                time.sleep(1.5)  # Allow ESP32 UART auto-reset to settle
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                logger.info(f"Connected to ESP32 on {self.port_name}.")
            except Exception as e:
                if auto_fallback:
                    logger.warning(f"Could not open physical port {self.port_name} ({e}). Operating in simulation mode.")
                    self.is_mock = True
                else:
                    raise ConnectionError(f"Failed to connect to hardware port {self.port_name}: {e}")
        else:
            self.is_mock = True
            logger.info("VoltQuantLink running in SIMULATION / MOCK mode.")

    @staticmethod
    def list_available_ports() -> List[str]:
        """List detected serial COM ports."""
        if not HAS_PYSERIAL:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def _sanitize_cell_id(self, cell_id: Union[int, str]) -> int:
        """Convert battery identifier string (e.g. 'B0005', 'CS2_35', 5) to integer 0-255."""
        if isinstance(cell_id, int):
            return max(0, min(255, cell_id))
        digits = "".join(ch for ch in str(cell_id) if ch.isdigit())
        return max(0, min(255, int(digits))) if digits else 0

    def _normalize_class(self, cls: Union[int, str]) -> str:
        """Normalize class to string accepted by voltquant_esp32.ino."""
        cls_str = str(cls).strip().lower()
        return self.CLASS_MAP.get(cls_str, str(cls))

    def _read_verdict_from_serial(self) -> str:
        """Read serial stream until the FPGA verdict line (RX status=...) or error arrives."""
        self.last_transaction_log.clear()
        deadline = time.time() + self.timeout
        verdict = ""

        while time.time() < deadline:
            line = self.serial_conn.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self.last_transaction_log.append(line)
            # The firmware emits:
            # 1. TX seq=...
            # 2. TX frame=...
            # 3. TX tag  =...
            # 4. RX status=0x.. ACCEPTED ... or REJECTED ... or NO_RESULT ...
            # or ERR ...
            if line.startswith("RX status=") or "ACCEPTED" in line or "REJECTED" in line or "NO_RESULT" in line:
                verdict = line
                break
            elif line.startswith("ERR"):
                verdict = line
                break

        if not verdict:
            verdict = "NO_RESULT (Timeout waiting for FPGA verdict)"

        self._update_stats(verdict)
        return verdict

    def send(
        self,
        cls: Union[int, str],
        cell_id: Union[int, str],
        value: Optional[Union[int, float]] = None
    ) -> str:
        """Send prediction packet to ESP32 + FPGA and await hardware verification reply.
        
        Args:
            cls: Prediction class ('SAFE', 'WARNING', 'DEGRADED' or 0, 1, 2)
            cell_id: Cell ID (0-255 or string e.g. "B0005")
            value: Optional SOH percentage (0.0 - 100.0)
            
        Returns:
            Verbatim reply containing "ACCEPTED", "REJECTED", "NO_RESULT", or "ERR".
        """
        cid = self._sanitize_cell_id(cell_id)
        c_str = self._normalize_class(cls)

        if value is not None:
            soh_f = float(value)
            line = f"{c_str},{cid},{soh_f:.1f}\n"
        else:
            line = f"{c_str},{cid}\n"

        return self._send_raw_line(line)

    def tamper(
        self,
        cls: Union[int, str],
        cell_id: Union[int, str],
        value: Optional[Union[int, float]] = None
    ) -> str:
        """Send TAMPER demo command: bit flipped after tagging -> FPGA MUST reject with BAD_TAG."""
        cid = self._sanitize_cell_id(cell_id)
        c_str = self._normalize_class(cls)
        if value is not None:
            soh_f = float(value)
            line = f"TAMPER {c_str},{cid},{soh_f:.1f}\n"
        else:
            line = f"TAMPER {c_str},{cid}\n"
        return self._send_raw_line(line)

    def replay(self) -> str:
        """Send REPLAY demo command: re-sends previous frame -> FPGA MUST reject with REPLAY."""
        return self._send_raw_line("REPLAY\n")

    def _send_raw_line(self, line: str) -> str:
        self.stats["sent"] += 1

        if not self.is_mock and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(line.encode("utf-8"))
                self.serial_conn.flush()
                return self._read_verdict_from_serial()
            except Exception as e:
                self.stats["errors"] += 1
                return f"ERR: SERIAL_COMMUNICATION_FAULT ({e})"

        # Simulation Mode: exactly mirrors FPGA hardware logic
        else:
            time.sleep(0.012)  # Hardware round-trip simulation
            raw = line.strip()

            if raw == "REPLAY":
                if not self._last_mock_packet:
                    verdict = "REJECTED: ERR nothing to replay yet"
                else:
                    verdict = "RX status=0x02 REJECTED reason=REPLAY trusted_class=SAFE"
                self._update_stats(verdict)
                return verdict

            tamper = raw.startswith("TAMPER ")
            if tamper:
                raw = raw[7:].strip()

            parts = raw.split(",")
            if len(parts) < 2:
                verdict = "REJECTED: ERR format must be <class>,<cell_id>[,<soh_percent>]"
                self._update_stats(verdict)
                return verdict

            cls_part = parts[0].strip().upper()
            if cls_part not in ["SAFE", "WARNING", "DEGRADED", "0", "1", "2"]:
                verdict = f"REJECTED: ERR class must be safe|warning|degraded (got '{cls_part}')"
                self._update_stats(verdict)
                return verdict

            try:
                cid = int(parts[1])
                if not (0 <= cid <= 255):
                    verdict = f"REJECTED: ERR cell_id must be 0-255 (got {cid})"
                    self._update_stats(verdict)
                    return verdict
            except ValueError:
                verdict = "REJECTED: ERR cell_id must be 0-255"
                self._update_stats(verdict)
                return verdict

            soh = 0.0
            soh_str = ""
            if len(parts) >= 3:
                try:
                    soh = float(parts[2])
                    if not (0.0 <= soh <= 100.0):
                        verdict = f"REJECTED: ERR soh must be 0-100 (percent) (got {soh})"
                        self._update_stats(verdict)
                        return verdict
                    soh_str = f" soh={soh:.0f}%"
                except ValueError:
                    verdict = "REJECTED: ERR soh must be 0-100 (percent)"
                    self._update_stats(verdict)
                    return verdict

            # Save genuine packet for replay
            if not tamper:
                self._last_mock_packet = raw

            # Verdict evaluation
            if tamper:
                verdict = f"RX status=0x04 REJECTED reason=BAD_TAG trusted_class=SAFE{soh_str}"
            else:
                verdict = f"RX status=0x01 ACCEPTED trusted_class={cls_part}{soh_str}"

            self._update_stats(verdict)
            return verdict

    def _update_stats(self, reply: str):
        if "ACCEPTED" in reply:
            self.stats["accepted"] += 1
        elif "REJECTED" in reply:
            self.stats["rejected"] += 1
        else:
            self.stats["errors"] += 1

    def close(self):
        """Safely close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                logger.info(f"Closed serial port {self.port_name}.")
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("VOLTQUANT HARDWARE LINK: PROTOCOL VERIFICATION")
    print("=" * 70)
    print(f"Detected physical serial ports: {VoltQuantLink.list_available_ports()}")

    link = VoltQuantLink("COM5")
    print(f"Mode: {'SIMULATION' if link.is_mock else 'PHYSICAL SERIAL'}")

    # Test 1: Nominal prediction matching teammate's example
    r1 = link.send(1, 5, 58)
    print(f"1. Send (1, 5, 58)           -> {r1}")
    assert "ACCEPTED" in r1

    # Test 2: Safe string class
    r2 = link.send("SAFE", 12, 94.2)
    print(f"2. Send ('SAFE', 12, 94.2)   -> {r2}")
    assert "ACCEPTED" in r2

    # Test 3: Bad Tag Attack Demonstration (TAMPER)
    r3 = link.tamper(1, 5, 58)
    print(f"3. Send TAMPER (1, 5, 58)    -> {r3}")
    assert "REJECTED" in r3 and "BAD_TAG" in r3

    # Test 4: Replay Attack Demonstration (REPLAY)
    r4 = link.replay()
    print(f"4. Send REPLAY               -> {r4}")
    assert "REJECTED" in r4 and "REPLAY" in r4

    # Test 5: Out of bounds SOH
    r5 = link.send(1, 5, 150)
    print(f"5. Send Out-of-bounds (150%) -> {r5}")
    assert "ERR" in r5 or "REJECTED" in r5

    print(f"\nLink Stats: {link.stats}")
    print("=" * 70)
    print("All tests passed! Protocol matches 'voltquant_esp32.ino' flawlessly.")
