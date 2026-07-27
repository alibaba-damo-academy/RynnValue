# adapted from openpi
"""Smoke test for the Franka env rollout server.

Connects to a running ``client/run_client.py`` instance and exercises every
WebSocket operation it exposes:

    - create_env
    - get_observation
    - reset
    - step  (with a safe zero / hold-current action)
    - get_info_for_step

By default it talks to ``ws://localhost:8101``. Override with
``--host`` / ``--port``.

Usage
-----
    # From the EXPO-FT project root (so that `expo_ft` is importable):
    cd EXPO-FT
    python scripts/franka/test_franka_client.py
    python scripts/franka/test_franka_client.py --host localhost --port 8101
    python scripts/franka/test_franka_client.py --num-steps 10 --skip-reset

Notes
-----
- ``reset()`` on the Franka env blocks until the human operator presses Enter
  on the *server* terminal (interactive manual reset). Pass ``--skip-reset``
  when you cannot interact with that terminal.
- The action sent in ``step`` is "hold current state": we read the current
  joint positions from the observation and send them back. This is the
  safest action you can publish to a real robot.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from typing import Any, Dict

import numpy as np
from PIL import Image

# Make sure we can import from the EXPO-FT project no matter where we run from.
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from expo_ft.env.env_client import EnvClient  # noqa: E402


# ─── Helpers ───────────────────────────────────────────────────────────

def _summarize_obs(obs: Dict[str, Any]) -> str:
    """Pretty-print the shapes / dtypes of an observation dict."""
    parts = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            parts.append(f"{k}: ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, (bytes, bytearray)):
            parts.append(f"{k}: bytes len={len(v)}")
        elif isinstance(v, str):
            preview = v if len(v) <= 60 else v[:57] + "..."
            parts.append(f"{k}: str={preview!r}")
        else:
            parts.append(f"{k}: {type(v).__name__}={v!r}")
    return "\n  ".join(parts)


def _save_obs_images(
    obs: Dict[str, Any],
    out_dir: str,
    tag: str,
) -> None:
    """Save the image/wrist_image fields of ``obs`` as PNG files under ``out_dir``.

    File names are ``<tag>_<key>.png`` (e.g. ``init_base_image.png``,
    ``step_03_left_wrist_image.png``).
    """
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("franka_client_test")
    for key in ("base_image", "left_wrist_image", "right_wrist_image"):
        img = obs.get(key)
        if not isinstance(img, np.ndarray):
            continue
        # Make sure it is uint8 RGB (H, W, 3).
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        path = os.path.join(out_dir, f"{tag}_{key}.png")
        Image.fromarray(arr).save(path)
        log.info("  saved %s -> %s (shape=%s)", key, path, arr.shape)


def _check_obs(obs: Dict[str, Any]) -> None:
    """Light validation that the observation has the expected Franka keys."""
    required = {"base_image", "left_wrist_image", "right_wrist_image", "state", "prompt"}
    missing = required - set(obs.keys())
    if missing:
        raise AssertionError(f"Observation is missing keys: {sorted(missing)}")

    for img_key in ("base_image", "left_wrist_image", "right_wrist_image"):
        img = obs[img_key]
        if not isinstance(img, np.ndarray):
            raise AssertionError(f"{img_key} should be ndarray, got {type(img)}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise AssertionError(
                f"{img_key} should be (H, W, 3), got shape {img.shape}"
            )
        if img.dtype != np.uint8:
            raise AssertionError(
                f"{img_key} should be uint8, got dtype {img.dtype}"
            )

    state = obs["state"]
    if not isinstance(state, np.ndarray):
        raise AssertionError(f"state should be ndarray, got {type(state)}")
    if state.ndim != 1 or state.shape[0] not in (8, 32):
        raise AssertionError(
            f"state should be 1-D of dim 32 (dual arm) or 8 (single arm), got shape {state.shape}"
        )


# ─── Test steps ────────────────────────────────────────────────────────

def _logical_joint_to_action_idx(logical_joint: int, arm_mode: str) -> int:
    """Map a logical joint index to the actual action array index.

    In the interleaved action format:
      - Dual arm: [left_joints(7), left_grip(1), right_joints(7), right_grip(1)]
        logical joint 0-6 → action index 0-6 (left arm)
        logical joint 7-13 → action index 8-14 (right arm, skip gripper at 7)
      - Single arm: [joints(7), grip(1)]
        logical joint 0-6 → action index 0-6
    """
    if arm_mode == "single":
        return logical_joint
    # Dual arm: left joints [0:7] -> action[0:7], right joints -> action[8:15]
    if logical_joint < 7:
        return logical_joint
    else:
        return logical_joint + 1  # skip left gripper at index 7


def _get_gripper_indices(arm_mode: str) -> list:
    """Return the action array indices that correspond to grippers.

    Interleaved format:
      - Dual arm: grippers at [7, 15]
      - Single arm: gripper at [7]
    """
    if arm_mode == "single":
        return [7]
    return [7, 15]


def _compute_action(
    state: np.ndarray,
    start_state: np.ndarray,
    action_dim: int,
    joint_dim: int,
    arm_mode: str,
    mode: str,
    step_idx: int,
    *,
    wiggle_joint: int,
    wiggle_amp: float,
    wiggle_period: int,
    gripper_target: float,
    gripper_period: int,
) -> np.ndarray:
    """Build the action vector for a given step according to ``mode``.

    Action format (interleaved, aligned with franka_client_sync.py):
      - Dual arm 16-dim: [left_joints(7), left_grip(1), right_joints(7), right_grip(1)]
      - Single arm 8-dim: [joints(7), grip(1)]

    All non-trivial modes are anchored to ``start_state`` (snapshotted right
    before the step loop) so that the commanded trajectory does not drift
    even if the controller has small tracking errors.

    Modes
    -----
    - ``hold``    : action = current state[:action_dim] (no motion).
    - ``wiggle``  : action = start_state with a sinusoidal offset added to
                    joint ``wiggle_joint``. Phase uses ``(step_idx + 1)`` so
                    the first step is already non-zero.
    - ``gripper`` : action = start_state, but grippers alternate between
                    their start value and ``gripper_target`` every
                    ``gripper_period`` steps.
    """
    if mode == "hold":
        return state[:action_dim].astype(np.float64).copy()

    a = start_state[:action_dim].astype(np.float64).copy()

    if mode == "wiggle":
        # Resolve negative index relative to the arm joints only.
        j = wiggle_joint if wiggle_joint >= 0 else (joint_dim + wiggle_joint)
        if not (0 <= j < joint_dim):
            raise ValueError(
                f"--wiggle-joint={wiggle_joint} resolves to {j}, "
                f"out of range [0, {joint_dim})."
            )
        # Map logical joint index to actual action array index (interleaved format)
        action_idx = _logical_joint_to_action_idx(j, arm_mode)
        # +1 so step 1 already has a non-zero offset.
        phase = 2.0 * np.pi * ((step_idx + 1) / max(1, wiggle_period))
        a[action_idx] = a[action_idx] + wiggle_amp * np.sin(phase)
        return a

    if mode == "gripper":
        toggle = (step_idx // max(1, gripper_period)) % 2 == 1
        if toggle:
            for gi in _get_gripper_indices(arm_mode):
                a[gi] = float(gripper_target)
        return a

    raise ValueError(f"Unknown action mode: {mode!r}")


def run_test(
    host: str,
    port: int,
    num_steps: int,
    skip_reset: bool,
    video_dir: str = "",
    env_usage: str = "eval",
    save_dir: str = "",
    action_mode: str = "hold",
    wiggle_joint: int = -1,
    wiggle_amp: float = 0.05,
    wiggle_period: int = 20,
    gripper_target: float = 0.0,
    gripper_period: int = 10,
    step_hz: float = 5.0,
    confirm: bool = True,
) -> None:
    log = logging.getLogger("franka_client_test")
    log.info("=" * 70)
    log.info("Franka env client smoke test")
    log.info("Target server: ws://%s:%d", host, port)
    log.info("=" * 70)

    client = EnvClient(host=host, port=port)

    env_id: str = ""
    try:
        # 1) create_env -----------------------------------------------------
        log.info("[1/5] create_env (env_usage=%s)...", env_usage)
        t0 = time.time()
        env_id, task_description = client.create_env(
            {"env_usage": env_usage, "video_dir": video_dir}
        )
        log.info(
            "  -> env_id=%r task=%r (%.2fs)",
            env_id, task_description, time.time() - t0,
        )

        # 2) get_observation ------------------------------------------------
        log.info("[2/5] get_observation...")
        t0 = time.time()
        obs = client.get_observation(env_id)
        log.info("  -> received in %.2fs", time.time() - t0)
        log.info("  observation:\n  %s", _summarize_obs(obs))
        _check_obs(obs)
        if save_dir:
            _save_obs_images(obs, save_dir, tag="init")
        state = np.asarray(obs["state"], dtype=np.float64)
        state_dim = int(state.shape[0])
        # State is 32-dim (dual arm) or 8-dim (single arm)
        # Action is 16-dim (dual) or 8-dim (single) — interleaved format
        if state_dim == 32:
            arm_mode = "dual"
            action_dim = 16
            joint_dim = 14
        else:
            arm_mode = "single"
            action_dim = 8
            joint_dim = 7
        log.info("  detected arm mode: %s (state_dim=%d, action_dim=%d, joint_dim=%d)",
                 arm_mode, state_dim, action_dim, joint_dim)

    # 3) reset ----------------------------------------------------------
        if skip_reset:
            log.info("[3/5] reset SKIPPED (--skip-reset).")
        else:
            log.info(
                "[3/5] reset... (server may block waiting for operator to press "
                "Enter on its terminal)"
            )
            t0 = time.time()
            obs, done = client.reset(env_id)
            log.info("  -> reset done=%s (%.2fs)", done, time.time() - t0)
            _check_obs(obs)
            if save_dir:
                _save_obs_images(obs, save_dir, tag="reset")
            state = np.asarray(obs["state"], dtype=np.float64)

        # 4) step ------------------------------------------------------------
        log.info(
            "[4/5] step x %d (action_mode=%s, step_hz=%.1f)",
            num_steps, action_mode, step_hz,
        )

        if action_mode == "wiggle":
            j_resolved = wiggle_joint if wiggle_joint >= 0 else joint_dim + wiggle_joint
            action_idx = _logical_joint_to_action_idx(j_resolved, arm_mode)
            log.info(
                "  wiggle: logical_joint=%d action_idx=%d amp=%.4f rad period=%d steps",
                j_resolved, action_idx, wiggle_amp, wiggle_period,
            )
        elif action_mode == "gripper":
            gripper_idxs = _get_gripper_indices(arm_mode)
            log.info(
                "  gripper: target=%.4f period=%d steps (toggle each period) "
                "gripper_indices=%s",
                gripper_target, gripper_period, gripper_idxs,
            )

        if action_mode != "hold" and confirm:
            log.warning("!!! ROBOT WILL MOVE. Press Enter to start, Ctrl-C to abort.")
            try:
                input()
            except EOFError:
                log.info("  no TTY: continuing without confirmation.")

        step_dt = 1.0 / step_hz if step_hz > 0 else 0.0
        # Make full-vector printing readable.
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        # Snapshot starting pose so wiggle/gripper trajectories are anchored
        # rather than drifting with the latest observation.
        start_state = state.copy()
        log.info("    start state: %s", np.array2string(start_state))

        # Offline sanity check: dry-run _compute_action for the first few
        # steps WITHOUT touching the robot. If these values do not vary
        # across steps, the parameters (mode/amp/period) themselves are
        # wrong; if they do vary but the on-robot loop below shows no
        # variation, something inside the loop is off.
        log.info("  --- _compute_action dry-run (no robot command) ---")
        for _i in range(min(5, num_steps)):
            _a = _compute_action(
                state=state,
                start_state=start_state,
                action_dim=action_dim,
                joint_dim=joint_dim,
                arm_mode=arm_mode,
                mode=action_mode,
                step_idx=_i,
                wiggle_joint=wiggle_joint,
                wiggle_amp=wiggle_amp,
                wiggle_period=wiggle_period,
                gripper_target=gripper_target,
                gripper_period=gripper_period,
            )
            log.info("    dry-run i=%d  action=%s  delta_from_start=%s",
                     _i,
                     np.array2string(_a),
                     np.array2string(_a - start_state[:action_dim]))
        log.info("  --- end dry-run ---")

        for i in range(num_steps):
            action = _compute_action(
                state=state,
                start_state=start_state,
                action_dim=action_dim,
                joint_dim=joint_dim,
                arm_mode=arm_mode,
                mode=action_mode,
                step_idx=i,
                wiggle_joint=wiggle_joint,
                wiggle_amp=wiggle_amp,
                wiggle_period=wiggle_period,
                gripper_target=gripper_target,
                gripper_period=gripper_period,
            )
            delta = action - state[:action_dim]
            delta_from_start = action - start_state[:action_dim]
            t0 = time.time()
            executed_action, action_type = client.step(env_id, action)
            dt = time.time() - t0
            executed_action = np.asarray(executed_action, dtype=np.float64)
            log.info(
                "  step %2d/%d: type=%s exec_dim=%d max|delta|=%.4g max|delta_from_start|=%.4g max|exec-cmd|=%.4g (%.2fs)",
                i + 1, num_steps, action_type, executed_action.shape[0],
                float(np.max(np.abs(delta))),
                float(np.max(np.abs(delta_from_start))),
                float(np.max(np.abs(executed_action - action[:executed_action.shape[0]]))),
                dt,
            )
            log.info("    sent     action: %s", np.array2string(action))
            log.info("    executed action: %s", np.array2string(executed_action))
            log.info("    delta (sent-state): %s", np.array2string(delta))
            log.info("    delta_from_start : %s", np.array2string(delta_from_start))
            # Refresh state from obs each step.
            obs = client.get_observation(env_id)
            _check_obs(obs)
            if save_dir:
                _save_obs_images(obs, save_dir, tag=f"step_{i + 1:02d}")
            new_state = np.asarray(obs["state"], dtype=np.float64)
            state_change = new_state - state[:new_state.shape[0]]
            state_drift = new_state - start_state[:new_state.shape[0]]
            log.info("    current  state : %s", np.array2string(new_state))
            log.info("    state-prev     : %s (max|.|=%.4g)",
                     np.array2string(state_change),
                     float(np.max(np.abs(state_change))))
            log.info("    state_from_start: %s (max|.|=%.4g)",
                     np.array2string(state_drift),
                     float(np.max(np.abs(state_drift))))
            state = new_state
            # Pace the loop so motion is smooth and the controller can keep up.
            sleep_left = step_dt - (time.time() - t0)
            if sleep_left > 0:
                time.sleep(sleep_left)

        # 5) get_info_for_step ----------------------------------------------
        log.info("[5/5] get_info_for_step...")
        t0 = time.time()
        done, success, reward, mask = client.get_info_for_step(env_id)
        log.info(
            "  -> done=%s success=%s reward=%.3f mask=%.3f (%.2fs)",
            done, success, reward, mask, time.time() - t0,
        )

        log.info("=" * 70)
        log.info("ALL CHECKS PASSED ✔")
        log.info("=" * 70)
    finally:
        # Always notify the server so it tears down the env (and its ROS2
        # spin thread) before we drop the WebSocket. Otherwise the next
        # create_env on the same env_id will collide with the lingering
        # spin thread ("generator already executing").
        if env_id:
            log.info("Closing env %s on server...", env_id)
            client.close_env(env_id)
        client.close()


# ─── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for the Franka env rollout server."
    )
    parser.add_argument("--host", default="localhost",
                        help="Rollout server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8101,
                        help="Rollout server port (default: 8101)")
    parser.add_argument("--num-steps", type=int, default=3,
                        help="How many step() calls to issue (default: 3)")
    parser.add_argument("--skip-reset", action="store_true",
                        help="Skip reset() (which on the real robot blocks "
                             "for manual confirmation on the server terminal).")
    parser.add_argument("--video-dir", default="",
                        help="Optional video_dir passed to create_env.")
    parser.add_argument("--env-usage", default="eval",
                        choices=["train", "eval"],
                        help="env_usage tag passed to create_env (default: eval).")
    parser.add_argument("--save-dir", default="./franka_obs_dump",
                        help="Directory to save observation images as PNG. "
                             "Pass an empty string to disable saving. "
                             "(default: ./franka_obs_dump)")
    parser.add_argument("--action-mode", default="hold",
                        choices=["hold", "wiggle", "gripper"],
                        help="hold = no motion (safe default); "
                             "wiggle = sinusoidal offset on one joint; "
                             "gripper = toggle gripper(s) between current and --gripper-target.")
    parser.add_argument("--wiggle-joint", type=int, default=-1,
                        help="Joint index to wiggle. Negative = from end of arm "
                             "joints (-1 = last arm joint, e.g. joint7). "
                             "For dual-arm, indices [0..6] are left arm, [7..13] are right.")
    parser.add_argument("--wiggle-amp", type=float, default=0.05,
                        help="Wiggle amplitude in radians (default: 0.05 ≈ 2.9°).")
    parser.add_argument("--wiggle-period", type=int, default=20,
                        help="Wiggle period in steps (default: 20).")
    parser.add_argument("--gripper-target", type=float, default=0.0,
                        help="Target gripper position to toggle to (default: 0.0).")
    parser.add_argument("--gripper-period", type=int, default=10,
                        help="Toggle gripper every N steps (default: 10).")
    parser.add_argument("--step-hz", type=float, default=5.0,
                        help="Step rate (Hz). Lower = safer/slower (default: 5.0).")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the 'press Enter to start motion' confirmation.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        run_test(
            host=args.host,
            port=args.port,
            num_steps=args.num_steps,
            skip_reset=args.skip_reset,
            video_dir=args.video_dir,
            env_usage=args.env_usage,
            save_dir=args.save_dir,
            action_mode=args.action_mode,
            wiggle_joint=args.wiggle_joint,
            wiggle_amp=args.wiggle_amp,
            wiggle_period=args.wiggle_period,
            gripper_target=args.gripper_target,
            gripper_period=args.gripper_period,
            step_hz=args.step_hz,
            confirm=not args.yes,
        )
    except AssertionError as e:
        logging.error("ASSERTION FAILED: %s", e)
        return 2
    except Exception as e:  # noqa: BLE001
        logging.error("TEST FAILED: %s", e)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
