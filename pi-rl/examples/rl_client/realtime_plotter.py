# adapted from openpi
"""Real-time plotting module: draws gripper action curves in a separate process.

Usage:
    from realtime_plotter import start_plotter, stop_plotter, send_plot_data
"""

import os
import time
from multiprocessing import Process, Queue

import numpy as np


def _atomic_savefig(fig, path: str, dpi: int = 100):
    """Save figure atomically: write to .tmp then rename, so readers never see a 0-byte file."""
    tmp = path + ".tmp"
    ext = os.path.splitext(path)[1].lstrip(".") or "png"
    fig.savefig(tmp, dpi=dpi, format=ext)
    os.replace(tmp, path)


def _plotter_process(data_queue: Queue, plot_dir: str):
    """Separate process: plots gripper action curves in real time.

    Receives data via a Queue and refreshes the display window with plt.ion().
    Falls back to periodically saving images automatically on headless environments.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Try an interactive backend; fall back to Agg on failure
    has_display = True
    try:
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure()  # test whether a window can be created
        plt.close()
    except Exception:
        has_display = False
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # state_history: one point per inference call
    # action_history: flattened action sequence (all horizon steps concatenated)
    # gt_action_history: dataset ground-truth action (only in eval_on_dataset), same x as state
    # state_x: positions of state / gt_action on the action-step time axis
    state_history = []   # list of (D_state,)
    state_x = []         # list of int
    action_history = []  # list of (D_action,)
    gt_action_history = []  # list of (D_action,) or None
    total_action_steps = 0  # cumulative number of action steps

    # Initialize figure
    fig = None
    axes = None
    lines_action = None
    lines_action_r = None
    action_dim = 8

    def _init_figure(adim):
        nonlocal fig, axes, lines_action, lines_action_r, action_dim
        action_dim = adim
        # 8 subplots: joint1~7 + gripper
        n_rows = 8
        fig, axes_arr = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows), sharex=True)
        axes = list(axes_arr)
        fig.suptitle("Real-time curves (action vs state)", fontsize=12)

        lines_action = {}

        # Joint 1~7 subplots
        for j in range(7):
            ax = axes[j]
            lines_action[f"j{j}"] = []
            ln, = ax.plot([], [], label="pred_action", linewidth=1.5, color="tab:blue")
            lines_action[f"j{j}"].append(ln)
            ln, = ax.plot([], [], marker='.', markersize=3, linestyle='',
                          label="state", color="tab:orange")
            lines_action[f"j{j}"].append(ln)
            ln, = ax.plot([], [], marker='x', markersize=4, linestyle='',
                          label="gt_action", color="tab:green")
            lines_action[f"j{j}"].append(ln)
            ax.set_ylabel(f"J{j+1} (rad)", fontsize=8)
            ax.set_title(f"Joint {j+1}", fontsize=9, loc="left")
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)

        # Gripper subplot
        ax = axes[7]
        lines_action["gripper"] = []
        ln, = ax.plot([], [], label="pred_action", linewidth=1.5, color="tab:blue")
        lines_action["gripper"].append(ln)
        ln, = ax.plot([], [], marker='.', markersize=3, linestyle='',
                      label="state", color="tab:orange")
        lines_action["gripper"].append(ln)
        ln, = ax.plot([], [], marker='x', markersize=4, linestyle='',
                      label="gt_action", color="tab:green")
        lines_action["gripper"].append(ln)
        ax.axhline(y=200, color='red', linestyle=':', alpha=0.5, label="thr=200")
        ax.set_ylabel("Grip (mm)", fontsize=8)
        ax.set_title("Gripper", fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)

        lines_action_r = None
        axes[-1].set_xlabel("Action Step")
        plt.tight_layout()

    def _update_figure():
        if len(action_history) < 2:
            return
        states = np.array(state_history)
        actions = np.array(action_history)
        T_act = actions.shape[0]
        t_act = np.arange(T_act)
        sx = np.array(state_x)

        gt_mask = np.array([g is not None for g in gt_action_history])
        if gt_mask.any():
            gt_arr = np.array([g for g in gt_action_history if g is not None])
            gt_x = sx[gt_mask]
        else:
            gt_arr = None
            gt_x = None

        # Joint 1~7
        for j in range(7):
            lines_action[f"j{j}"][0].set_data(t_act, actions[:, j])
            lines_action[f"j{j}"][1].set_data(sx, states[:, j])
            if gt_arr is not None:
                lines_action[f"j{j}"][2].set_data(gt_x, gt_arr[:, j])
            axes[j].set_xlim(0, max(T_act, 10))
            axes[j].relim()
            axes[j].autoscale_view(scalex=False)

        # Gripper
        lines_action["gripper"][0].set_data(t_act, actions[:, 7])
        lines_action["gripper"][1].set_data(sx, states[:, 7])
        if gt_arr is not None:
            lines_action["gripper"][2].set_data(gt_x, gt_arr[:, 7])
        axes[7].set_xlim(0, max(T_act, 10))
        axes[7].relim()
        axes[7].autoscale_view(scalex=False)

    # Main loop: read data from the queue and refresh in real time
    save_counter = 0
    while True:
        try:
            # Non-blocking read of all pending data
            got_data = False
            while not data_queue.empty():
                msg = data_queue.get_nowait()
                if msg is None:  # stop signal
                    # Save the final plot
                    if fig is not None and len(action_history) >= 2:
                        _update_figure()
                        final_path = os.path.join(plot_dir, "curves_final.png")
                        latest_path = os.path.join(plot_dir, "curves_latest.png")
                        _atomic_savefig(fig, final_path, dpi=150)
                        _atomic_savefig(fig, latest_path, dpi=150)
                        print(f"[Plotter] Final curves saved: {final_path}")
                    return
                state, action_chunk, gt_action = msg
                # action_chunk: (horizon, D) — expand all steps
                action_chunk = np.asarray(action_chunk)
                horizon = action_chunk.shape[0]
                # state / gt_action: record one point per inference; x aligned to the start of the current action steps
                state_history.append(state)
                state_x.append(total_action_steps)
                gt_action_history.append(gt_action)
                for step in range(horizon):
                    action_history.append(action_chunk[step])
                total_action_steps += horizon
                got_data = True

                # Initialize the figure on first data received
                if fig is None:
                    _init_figure(action_chunk.shape[1])

            if not got_data:
                time.sleep(0.05)
                continue

            # Refresh the chart
            _update_figure()

            if has_display:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.01)
            else:
                # Headless: atomically save the latest snapshot on every update
                save_counter += 1
                save_path = os.path.join(plot_dir, "curves_latest.png")
                _atomic_savefig(fig, save_path, dpi=100)

        except Exception as e:
            print(f"[Plotter] Error: {e}")
            time.sleep(0.1)


# ─── Public API ───────────────────────────────────────────────────────

def start_plotter(plot_dir: str = "plots") -> tuple[Queue, Process]:
    """Start the real-time plotting process; returns (data_queue, process)."""
    os.makedirs(plot_dir, exist_ok=True)
    data_queue = Queue(maxsize=500)
    proc = Process(
        target=_plotter_process,
        args=(data_queue, plot_dir),
        daemon=True,
    )
    proc.start()
    return data_queue, proc


def send_plot_data(
    queue: Queue,
    state: np.ndarray,
    action_chunk: np.ndarray,
    gt_action: np.ndarray | None = None,
):
    """Send one frame of data to the plotting process (non-blocking; skipped if the queue is full).

    Args:
        queue: data_queue returned by start_plotter
        state: (D_state,) current observation
        action_chunk: (horizon, D_action) raw model output
        gt_action: (D_action,) dataset ground-truth action, optional
    """
    try:
        gt_copy = gt_action.copy() if gt_action is not None else None
        queue.put_nowait((state.copy(), action_chunk.copy(), gt_copy))
    except Exception:
        pass


def stop_plotter(queue: Queue, proc: Process, timeout: float = 5.0):
    """Signal the plotting process to stop and wait for it to exit."""
    try:
        queue.put(None)
    except Exception:
        pass
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
