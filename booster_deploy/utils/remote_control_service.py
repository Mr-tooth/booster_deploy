"""Remote control service with joystick-first and keyboard fallback inputs."""

from __future__ import annotations

import atexit
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from types import TracebackType

import evdev


@dataclass
class JoystickConfig:
    """Store joystick axis/button mappings and scaling bounds."""

    max_vx: float = 1.0
    max_vy: float = 1.0
    max_vyaw: float = 1.0
    control_threshold: float = 0.1
    custom_mode_button: int = evdev.ecodes.BTN_A
    rl_gait_button: int = evdev.ecodes.BTN_B
    x_axis: int = evdev.ecodes.ABS_Y
    y_axis: int = evdev.ecodes.ABS_X
    yaw_axis: int = evdev.ecodes.ABS_Z


class RemoteControlService:
    """Read velocity commands from joystick events or keyboard input."""

    def __init__(self, config: JoystickConfig | None = None) -> None:
        """Initialize remote control service and background polling thread.

        Args:
            config: Optional joystick button/axis configuration.

        """
        self.config = config or JoystickConfig()
        self._lock = threading.Lock()
        self._running = True
        self.joystick: evdev.InputDevice | None = None
        self.joystick_runner: threading.Thread | None = None
        self.keyboard_runner: threading.Thread | None = None
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False
        self._stdin_tty = False
        self._old_termios: list[int] | None = None
        self.axis_ranges: dict[int, evdev.AbsInfo] = {}

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        try:
            self._init_joystick()
            self._start_joystick_thread()
        except Exception as exc:
            print(f"{exc}, downgrade to keyboard control")
            self._init_keyboard_control()
            self._start_keyboard_thread()

    def _has_joystick(self) -> bool:
        """Return whether a valid joystick device is currently connected."""
        return self.joystick is not None

    def get_operation_hint(self) -> str:
        """Return control instructions for the currently active input mode."""
        if self._has_joystick():
            return (
                "Joystick left axis for forward/backward/left/right, "
                "right axis for rotation left/right"
            )
        return (
            "Press keyboard 'w'/'s' to increase/decrease vx; Press 'a'/'d' to "
            "increase/decrease vy; Press 'q'/'e' to increase/decrease vyaw, "
            "press 'Space' to stop."
        )

    def get_custom_mode_operation_hint(self) -> str:
        """Return instruction for toggling custom mode."""
        if self._has_joystick():
            return "Press joystick button X to start custom mode."
        return "Press keyboard 'x' to start custom mode."

    def get_rl_gait_operation_hint(self) -> str:
        """Return instruction for toggling RL gait mode."""
        if self._has_joystick():
            return "Press joystick button A to start rl Gait."
        return "Press keyboard 'r' to start rl Gait."

    def _init_keyboard_control(self) -> None:
        """Initialize internal state for keyboard fallback control."""
        self.joystick = None
        self.joystick_runner = None
        self.keyboard_start_custom_mode = False
        self.keyboard_start_rl_gait = False

    def _start_keyboard_thread(self) -> None:
        """Start the keyboard listener thread and setup terminal recovery."""
        self.keyboard_runner = threading.Thread(
            target=self._keyboard_listener,
            daemon=True,
        )
        try:
            if sys.stdin.isatty():
                self._stdin_tty = True
                self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            else:
                self._stdin_tty = False
                self._old_termios = None
        except Exception:
            self._stdin_tty = False
            self._old_termios = None

        try:
            atexit.register(self.close)
        except Exception:
            pass

        self.keyboard_runner.start()

    def _keyboard_listener(self) -> None:
        """Read keyboard input in cbreak mode and translate it to commands."""
        fd: int | None = None
        try:
            if not self._stdin_tty:
                return
            fd = sys.stdin.fileno()
            tty.setcbreak(fd)
            while self._running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue
                char = sys.stdin.read(1)
                if char == "\x03":
                    continue
                key = "space" if char == " " else char
                try:
                    self._handle_keyboard_press(key)
                except Exception:
                    pass
        finally:
            try:
                if fd is not None and self._old_termios is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _handle_keyboard_press(self, key: str) -> None:
        """Apply a keyboard keypress to velocity commands and mode flags.

        Args:
            key: Single-character key token or `"space"`.

        """
        if key == "x":
            self.keyboard_start_custom_mode = True
        if key == "r":
            self.keyboard_start_rl_gait = True
        if key == "w":
            old_x = self.vx
            self.vx = min(self.vx + 0.1, self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "s":
            old_x = self.vx
            self.vx = max(self.vx - 0.1, -self.config.max_vx)
            print(f"VX: {old_x:.1f} => {self.vx:.1f}")
        if key == "a":
            old_y = self.vy
            self.vy = min(self.vy + 0.1, self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "d":
            old_y = self.vy
            self.vy = max(self.vy - 0.1, -self.config.max_vy)
            print(f"VY: {old_y:.1f} => {self.vy:.1f}")
        if key == "q":
            old_yaw = self.vyaw
            self.vyaw = min(self.vyaw + 0.1, self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "e":
            old_yaw = self.vyaw
            self.vyaw = max(self.vyaw - 0.1, -self.config.max_vyaw)
            print(f"VYaw: {old_yaw:.1f} => {self.vyaw:.1f}")
        if key == "space":
            self.vx = 0.0
            self.vy = 0.0
            self.vyaw = 0.0
            print("FULL STOP")

    def _init_joystick(self) -> None:
        """Detect and initialize a joystick device matching configured axes.

        Raises:
            RuntimeError: If no suitable joystick is available.

        """
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        joystick: evdev.InputDevice | None = None

        for device in devices:
            caps = device.capabilities()
            if evdev.ecodes.EV_ABS not in caps or evdev.ecodes.EV_KEY not in caps:
                continue

            abs_info = caps.get(evdev.ecodes.EV_ABS, [])
            axes = [code for (code, _info) in abs_info]
            if not all(code in axes for code in [self.config.x_axis, self.config.y_axis, self.config.yaw_axis]):
                continue

            absinfo: dict[int, evdev.AbsInfo] = {}
            for code, info in abs_info:
                absinfo[int(code)] = info
            self.axis_ranges = {
                self.config.x_axis: absinfo[self.config.x_axis],
                self.config.y_axis: absinfo[self.config.y_axis],
                self.config.yaw_axis: absinfo[self.config.yaw_axis],
            }
            joystick = device
            print(f"Found suitable joystick: {device.name}")
            break

        if joystick is None:
            raise RuntimeError("No suitable joystick found")

        self.joystick = joystick
        print(f"Selected joystick: {joystick.name}")

    def _start_joystick_thread(self) -> None:
        """Start a background thread that polls joystick events."""
        self.joystick_runner = threading.Thread(target=self._run_joystick, daemon=True)
        self.joystick_runner.start()

    def start_custom_mode(self) -> bool:
        """Return whether custom mode should be started at current step."""
        if self._has_joystick():
            assert self.joystick is not None
            return self.joystick.active_keys() == [self.config.custom_mode_button]
        return self.keyboard_start_custom_mode

    def start_rl_gait(self) -> bool:
        """Return whether RL gait mode should be started at current step."""
        if self._has_joystick():
            assert self.joystick is not None
            return self.joystick.active_keys() == [self.config.rl_gait_button]
        return self.keyboard_start_rl_gait

    def _run_joystick(self) -> None:
        """Poll joystick events and update velocity commands continuously."""
        while self._running:
            try:
                assert self.joystick is not None
                event = self.joystick.read_one()
                if event is None:
                    time.sleep(0.01)
                    continue
                if event.type == evdev.ecodes.EV_ABS:
                    self._handle_axis(int(event.code), int(event.value))
            except Exception as exc:
                if not self._running:
                    break
                print(f"Error in joystick polling loop: {exc}")
                time.sleep(0.05)

    def _handle_axis(self, code: int, value: int) -> None:
        """Handle joystick axis values and map them to velocity commands.

        Args:
            code: Axis code from evdev.
            value: Raw axis value.

        """
        if code == self.config.x_axis:
            self.vx = self._scale(value, self.config.max_vx, self.config.control_threshold, code)
        elif code == self.config.y_axis:
            self.vy = self._scale(value, self.config.max_vy, self.config.control_threshold, code)
        elif code == self.config.yaw_axis:
            self.vyaw = self._scale(value, self.config.max_vyaw, self.config.control_threshold, code)

    def _scale(
        self,
        value: float,
        max_value: float,
        threshold: float,
        axis_code: int,
    ) -> float:
        """Scale raw joystick axis value to normalized command value.

        Args:
            value: Raw axis value.
            max_value: Maximum absolute command value.
            threshold: Dead-zone threshold around zero.
            axis_code: Axis code used to lookup absolute range.

        Returns:
            Scaled command value.

        """
        absinfo = self.axis_ranges[axis_code]
        min_in = absinfo.min
        max_in = absinfo.max
        mapped_value = ((value - min_in) / (max_in - min_in) * 2 - 1) * max_value
        if abs(mapped_value) < threshold:
            return 0.0
        return -mapped_value

    def get_vx_cmd(self) -> float:
        """Return current forward velocity command."""
        with self._lock:
            return self.vx

    def get_vy_cmd(self) -> float:
        """Return current lateral velocity command."""
        with self._lock:
            return self.vy

    def get_vyaw_cmd(self) -> float:
        """Return current yaw velocity command."""
        with self._lock:
            return self.vyaw

    def close(self) -> None:
        """Stop polling threads and release joystick/terminal resources."""
        self._running = False
        try:
            if self._stdin_tty and self._old_termios is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass

        if self.joystick is not None:
            try:
                self.joystick.close()
            except Exception as exc:
                print(f"Error closing joystick: {exc}")

        if self.joystick_runner is not None:
            try:
                self.joystick_runner.join(timeout=1.0)
                if self.joystick_runner.is_alive():
                    print("Joystick thread didn't exit within the time limit")
            except Exception as exc:
                print(f"Error waiting for joystick thread to end: {exc}")

        if self.keyboard_runner is not None:
            try:
                self.keyboard_runner.join(timeout=1.0)
                if self.keyboard_runner.is_alive():
                    print("Keyboard thread didn't exit within the time limit")
            except Exception as exc:
                print(f"Error waiting for keyboard thread to end: {exc}")

    def __enter__(self) -> RemoteControlService:
        """Return context-manager instance."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the service on context-manager exit."""
        self.close()
