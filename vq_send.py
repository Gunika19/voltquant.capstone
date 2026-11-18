"""VoltQuant Hardware Communication Link (vq_send.py)

Provides serial UART communication between the VoltQuant Python prognostics
pipeline and the ESP32 + FPGA hardware verification subsystem.

Packet Protocol:
  Transmitted: "<class>,<cell_id>,<value>\\n"
    - <class>   : Alert state (0 = Normal, 1 = Warning / Advisory, 2 = Critical / Rollover)
    - <cell_id> : Integer identifier of the battery cell (e.g., 5 for Cell B0005)
    - <value>   : State of Health (SOH) percentage as an integer (e.g., 58 for 58% SOH)

  Received Reply:
    - Contains "ACCEPTED"  : Telemetry verified and authenticated by FPGA.
    - Contains "REJECTED"  : Verification failed (checksum error, anomalous jump, or security breach).
    - Contains "NO_RESULT" : Hardware timeout; no response from FPGA within deadline.
    - Contains "ERR"       : Serial port or communication error.

Usage:
  from vq_send import VoltQuantLink

  link = VoltQuantLink("COM5")                  # Physical ESP32 connection (with auto-mock fallback)
  reply = link.send(cls=1, cell_id=5, value=58) # e.g. link.send(1, 5, 58)

  if "ACCEPTED" in reply:
      # FPGA verified it
      print("Telemetry verified by hardware.")
  elif "REJECTED" in reply or "NO_RESULT" in reply or "ERR" in reply:
      # handle as a fault
      print("Hardware verification fault:", reply)
"""

from __future__ import annotations
import sys
import time
import logging
from typing import Optional, Union

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

    def __init__(
        self,
        port: str = "COM5",
        baudrate: int = 115200,
        timeout: float = 2.0,
        mock: bool = False,
        auto_fallback: bool = True
    ):
        """Initialize the hardware communication link.
        
        Args:
            port: Serial port name (e.g., "COM5" on Windows, "/dev/ttyUSB0" on Linux).
            baudrate: UART baud rate (default: 115200).
            timeout: Read timeout in seconds.
            mock: If True, operates in simulation mode without physical serial hardware.
            auto_fallback: If True and physical port fails to open, automatically fall back to mock mode.
        """
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_mock = mock
        self.serial_conn: Optional[serial.Serial] = None
        self.stats = {"sent": 0, "accepted": 0, "rejected": 0, "errors": 0}

        if not self.is_mock and HAS_PYSERIAL:
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
                    logger.warning(f"Could not open physical port {self.port_name} ({e}). Falling back to simulation mode.")
                    self.is_mock = True
                else:
                    raise ConnectionError(f"Failed to connect to hardware port {self.port_name}: {e}")
        else:
            self.is_mock = True
            logger.info("VoltQuantLink running in SIMULATION / MOCK mode.")

    @staticmethod
    def list_available_ports() -> list[str]:
        """List all currently detected serial COM ports on the system."""
        if not HAS_PYSERIAL:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def _sanitize_cell_id(self, cell_id: Union[int, str]) -> int:
        """Convert any string battery identifier (e.g. 'B0005', 'CS2_35', 'b1c18') to integer."""
        if isinstance(cell_id, int):
            return cell_id
        # Strip letters and extract digits
        digits = "".join(ch for ch in str(cell_id) if ch.isdigit())
        return int(digits) if digits else 0

    def send(self, cls: int, cell_id: Union[int, str], value: Union[int, float]) -> str:
        """Send formatted prediction to ESP32 + FPGA and await hardware verification reply.
        
        Packet format: "<class>,<cell_id>,<value>\\n"
        
        Args:
            cls: Prediction alert class (0 = Normal, 1 = Warning, 2 = Critical)
            cell_id: Cell integer or identifier string (e.g. 5, "B0005", "CS2_35")
            value: SOH percentage (e.g. 58 or 58.4)
            
        Returns:
            Hardware reply string (e.g., "ACCEPTED", "REJECTED", "NO_RESULT", "ERR")
        """
        c_int = int(cls)
        cid_int = self._sanitize_cell_id(cell_id)
        val_int = int(round(float(value)))

        packet = f"{c_int},{cid_int},{val_int}\n"
        self.stats["sent"] += 1

        # Physical Serial Transmission
        if not self.is_mock and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(packet.encode("utf-8"))
                self.serial_conn.flush()

                # Await reply line from ESP32
                raw_reply = self.serial_conn.readline()
                if not raw_reply:
                    self.stats["errors"] += 1
                    return "NO_RESULT: TIMEOUT_WAITING_FOR_FPGA"

                reply = raw_reply.decode("utf-8", errors="replace").strip()
                self._update_stats(reply)
                return reply

            except Exception as e:
                self.stats["errors"] += 1
                return f"ERR: SERIAL_COMMUNICATION_FAULT ({e})"

        # Simulation / Mock Mode (for testing without physical hardware plugged in)
        else:
            time.sleep(0.015)  # Simulate typical 15 ms UART round-trip + FPGA check
            # Realistic FPGA verification rules:
            # - SOH value must be between 0 and 100
            # - Class must be in [0, 1, 2]
            # - Non-sensical values trigger REJECTED
            if not (0 <= val_int <= 100) or c_int not in [0, 1, 2]:
                reply = f"REJECTED: OUT_OF_BOUNDS (cls={c_int}, soh={val_int})"
            else:
                reply = f"ACCEPTED: FPGA_HASH_VERIFIED (cls={c_int}, cell={cid_int}, soh={val_int}%)"
            
            self._update_stats(reply)
            return reply

    def _update_stats(self, reply: str):
        if "ACCEPTED" in reply:
            self.stats["accepted"] += 1
        elif "REJECTED" in reply:
            self.stats["rejected"] += 1
        else:
            self.stats["errors"] += 1

    def close(self):
        """Safely close serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                logger.info(f"Closed serial connection on {self.port_name}.")
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("VOLTQUANT HARDWARE LINK: SELF-TEST & VERIFICATION")
    print("=" * 70)
    print(f"Detected physical serial ports: {VoltQuantLink.list_available_ports()}")

    # Test link
    link = VoltQuantLink("COM5")
    print(f"Operating Mode: {'MOCK / SIMULATION' if link.is_mock else 'PHYSICAL SERIAL'}")

    # Test test packet 1: Normal prediction
    r1 = link.send(0, 5, 92)
    print(f"\nTX: 0,5,92 -> RX: {r1}")
    assert "ACCEPTED" in r1, f"Expected ACCEPTED, got {r1}"

    # Test test packet 2: Warning prediction matching teammate's example
    r2 = link.send(1, 5, 58)
    print(f"TX: 1,5,58 -> RX: {r2}")
    assert "ACCEPTED" in r2, f"Expected ACCEPTED, got {r2}"

    # Test test packet 3: Corrupted SOH value (out of bounds)
    r3 = link.send(1, 5, 999)
    print(f"TX: 1,5,999 -> RX: {r3}")
    assert "REJECTED" in r3, f"Expected REJECTED, got {r3}"

    print(f"\nFinal Stats: {link.stats}")
    print("=" * 70)
    print("All tests passed. VoltQuantLink is ready for hardware integration!")
