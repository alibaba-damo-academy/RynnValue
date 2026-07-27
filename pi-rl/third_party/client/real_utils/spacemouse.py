""" Test the spacemouse output. """
import threading
import time
import numpy as np
import pyspacemouse
from typing import Tuple

class SpaceMousePolicy:
    def __init__(self, max_lin_vel=1, max_rot_vel=1):
        self.movement_enabled = False
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.spacemouse = SpaceMouseExpert()
        # One-shot flags for keyboard A/B (e.g. calibration); set by GUI, read and cleared by get_info()
        self._virtual_success = False
        self._virtual_failure = False
        
    def get_action(self):
        return self.spacemouse.get_action()

    def forward(self, obs, include_info=False):
        """Forward pass: get action from spacemouse and return 7D action (6D velocities + 1D gripper).
        
        Args:
            obs: Observation dictionary (not used, but required for API compatibility)
            include_info: If True, return (action, info_dict), else return action only.
            
        Returns:
            If include_info=False: 7D numpy array [lin_vel(3), rot_vel(3), gripper(1)]
            If include_info=True: (action, info_dict) tuple
        """
        # Get raw action and buttons from spacemouse
        raw_action, buttons = self.spacemouse.get_action()
        
        # if have movement and movement_enabled is False, set movement_enabled to True
        if not self.movement_enabled and np.linalg.norm(raw_action[:-1]) > 0.0001:
            self.movement_enabled = True
        
        # Scale velocities
        lin_vel = raw_action[:3] * self.max_lin_vel
        rot_vel = raw_action[3:6] * self.max_rot_vel
        
        # Gripper control: button 0 typically controls gripper
        # 1.0 = open, -1.0 = close (matching VRPolicy convention)
        if len(buttons) > 0 and buttons[0]:
            gripper_vel = 1.0  # Close gripper
        else:
            gripper_vel = -1.0   # Open gripper
        
        # Concatenate to 7D action
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        
        # Clip to [-1, 1] range
        action = np.clip(action, -1.0, 1.0)
        if include_info:
            info_dict = {}
            return action, info_dict
        else:
            return action
    
    def set_virtual_success(self, value=True):
        """Set one-shot 'A' for calibration etc. Cleared on next get_info()."""
        self._virtual_success = bool(value)

    def set_virtual_failure(self, value=True):
        """Set one-shot 'B' for calibration etc. Cleared on next get_info()."""
        self._virtual_failure = bool(value)

    def get_info(self):
        success = self._virtual_success
        failure = self._virtual_failure
        self._virtual_success = False
        self._virtual_failure = False
        return {
            "movement_enabled": self.movement_enabled,
            "success": success,
            "failure": failure,
            "controller_on": True,
        }

    def reset_state(self):
        self.movement_enabled = False
        self._virtual_success = False
        self._virtual_failure = False

class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        # Pre-allocate arrays to avoid allocation inside lock
        self.latest_action = np.zeros(6, dtype=np.float64)
        self.latest_buttons = [0, 0]
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        """Continuously read spacemouse in background thread."""
        while True:
            state = pyspacemouse.read()
            if state is not None:
                # Create array outside lock to minimize lock time
                action = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
                    dtype=np.float64
                )  # spacemouse axis matched with robot base frame
                buttons = list(state.buttons) if hasattr(state.buttons, '__iter__') else [state.buttons]
                
                # Hold lock only for the minimal time needed to update
                with self.state_lock:
                    self.latest_action[:] = action  # In-place update, faster than assignment
                    self.latest_buttons = buttons
            # Small sleep to prevent excessive CPU usage and allow other threads to run
            time.sleep(0.001)  # 10ms sleep = ~100Hz max read rate

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        # Hold lock only for the minimal time needed to copy
        with self.state_lock:
            # Return copies to ensure thread safety
            return self.latest_action.copy(), self.latest_buttons.copy()


def test_spacemouse():
    """Test the SpaceMouseExpert class.

    This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
    The user is expected to move the spacemouse and press its buttons while the test is running.
    It keeps running until the user stops it.

    """
    spacemouse = SpaceMouseExpert()
    with np.printoptions(precision=3, suppress=True):
        while True:
            start_time = time.time()
            action, buttons = spacemouse.get_action()
            print(f"Spacemouse action: {action}, buttons: {buttons}")
            elapsed = time.time() - start_time
            print(f"Time taken to get action: {elapsed} seconds")
            time.sleep(1/15)  # 15Hz


def main():
    """Call spacemouse test."""
    test_spacemouse()


if __name__ == "__main__":
    main()