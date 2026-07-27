import io
import os
import shutil
import tempfile
import numpy as np
import imageio.v2 as imageio
import matplotlib

# Non-interactive backend to reduce matplotlib overhead/threading issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from robometer.data.datasets.helpers import load_frames_from_npz
from robometer.data.collators.utils import convert_frames_to_pil_images
from robometer.utils.logger import get_logger

logger = get_logger()

_DEFAULT_FONT = ImageFont.load_default()


def _fig_to_pil(fig, size=None):
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if size is not None:
            img = img.resize(size)
        return img
    finally:
        buf.close()
        plt.close(fig)


def _make_trend_plot(
    value,
    current_idx,
    baseline,
    size=(400, 300),
    title="Value Trend",
    task_title=None,
    baseline_label="baseline",
):
    w, h = size
    dpi = 100
    fig_w = max(w / dpi, 1.0)
    fig_h = max(h / dpi, 1.0)

    small_mode = (w < 320 or h < 260)
    medium_mode = (w < 420 or h < 320)

    if small_mode:
        title_fs = 9
        label_fs = 8
        tick_fs = 7
        legend_fs = 7
        line_w = 1.5
        marker_s1 = 20
        marker_s2 = 16
        show_legend = False
    elif medium_mode:
        title_fs = 10
        label_fs = 9
        tick_fs = 8
        legend_fs = 8
        line_w = 1.8
        marker_s1 = 24
        marker_s2 = 20
        show_legend = True
    else:
        title_fs = 12
        label_fs = 10
        tick_fs = 9
        legend_fs = 9
        line_w = 2.0
        marker_s1 = 30
        marker_s2 = 25
        show_legend = True

    value = np.asarray(value, dtype=float)
    baseline_curve = np.asarray(baseline, dtype=float)

    if len(baseline_curve) != len(value):
        raise ValueError(f"len(baseline)={len(baseline_curve)} != len(value)={len(value)}")

    x = np.arange(len(value))
    y = value

    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_margin = 0.05 * (y_max - y_min)

    b_min, b_max = float(np.min(baseline_curve)), float(np.max(baseline_curve))
    if b_min == b_max:
        b_min -= 1.0
        b_max += 1.0
    b_margin = 0.05 * (b_max - b_min)

    fig, ax1 = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=dpi,
        constrained_layout=True,
    )

    try:
        ax1.plot(
            x[: current_idx + 1],
            y[: current_idx + 1],
            color="tab:blue",
            linewidth=line_w,
            label="value",
        )
        ax1.scatter(
            [current_idx],
            [y[current_idx]],
            color="red",
            s=marker_s1,
            zorder=3,
            label="current value",
        )

        ax1.set_xlabel("Frame", fontsize=label_fs)
        ax1.set_ylabel("Value", color="tab:blue", fontsize=label_fs)
        ax1.tick_params(axis="x", labelsize=tick_fs)
        ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=tick_fs)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(len(value) - 1, 1))
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            baseline_curve,
            color="green",
            linestyle="--",
            linewidth=line_w,
            label=baseline_label,
        )
        ax2.scatter(
            [current_idx],
            [baseline_curve[current_idx]],
            color="green",
            s=marker_s2,
            zorder=3,
            label=f"current {baseline_label}",
        )
        ax2.set_ylabel(
            baseline_label if not small_mode else baseline_label[:12],
            color="green",
            fontsize=label_fs,
        )
        ax2.tick_params(axis="y", labelcolor="green", labelsize=tick_fs)
        ax2.set_ylim(b_min - b_margin, b_max + b_margin)

        full_title = title if (small_mode or task_title is None) else f"{task_title}\n{title}"
        ax1.set_title(full_title, fontsize=title_fs)

        if show_legend:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=legend_fs)

        return _fig_to_pil(fig, size=size)
    finally:
        plt.close(fig)


def _concat_horizontally(img1, img2, bg_color=(255, 255, 255)):
    h = max(img1.height, img2.height)
    w = img1.width + img2.width
    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    return canvas


def _draw_overlay_text(img, lines, xy=(10, 10), fill=(255, 0, 0), line_spacing=6, font=None):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    if font is None:
        font = _DEFAULT_FONT

    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        line_height = bbox[3] - bbox[1]
        y += line_height + line_spacing

    return img


def _ceil_to_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def _pad_image_to_multiple(img, multiple=16, bg_color=(255, 255, 255)):
    new_w = _ceil_to_multiple(img.width, multiple)
    new_h = _ceil_to_multiple(img.height, multiple)

    if new_w == img.width and new_h == img.height:
        return img

    canvas = Image.new("RGB", (new_w, new_h), bg_color)
    canvas.paste(img, (0, 0))
    return canvas


def save_video_with_trend(
    images,
    value,
    baseline,
    output_path,
    fps=30,
    plot_width_ratio=0.45,
    title="Value Trend",
    task_title=None,
    baseline_label="baseline",
    show_baseline_text=True,
    show_value_text=True,
    show_task_title=True,
    min_plot_width=260,
    max_plot_width=520,
    min_plot_height=220,
    codec="libx264",
    macro_block_size=16,
):
    if len(images) == 0:
        raise ValueError("images is empty")

    value = np.asarray(value, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    if len(value) == 0:
        raise ValueError("value is empty")
    if len(baseline) == 0:
        raise ValueError("baseline is empty")

    # Expand value/baseline to match images length via linear interpolation
    num_images = len(images)
    if len(value) != num_images:
        value = np.interp(
            np.linspace(0, len(value) - 1, num_images),
            np.arange(len(value)),
            value,
        )
    if len(baseline) != num_images:
        baseline = np.interp(
            np.linspace(0, len(baseline) - 1, num_images),
            np.arange(len(baseline)),
            baseline,
        )

    first_img = images[0].convert("RGB")
    base_w, base_h = first_img.size

    if base_h < min_plot_height:
        scale = float(min_plot_height) / float(base_h)
        base_w = int(round(base_w * scale))
        base_h = min_plot_height

    plot_w = int(base_w * plot_width_ratio)
    plot_w = max(min_plot_width, min(plot_w, max_plot_width))
    plot_h = max(min_plot_height, base_h)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to a local temp file first, then copy to the final destination.
    # Some network/shared filesystems (e.g. NFS-like mounts) do not support
    # the seek operations required to write the MP4 moov atom, which corrupts
    # the output. Writing locally avoids this entirely.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix=".plot_utils_")
    os.close(tmp_fd)
    try:
        with imageio.get_writer(
            tmp_path,
            fps=fps,
            codec=codec,
            macro_block_size=macro_block_size,
        ) as writer:
            for i, img in enumerate(images):
                img = img.convert("RGB")
                if img.size != (base_w, base_h):
                    img = img.resize((base_w, base_h))

                overlay_lines = []
                if show_task_title and task_title is not None:
                    overlay_lines.append(f"task: {task_title}")
                if show_value_text:
                    overlay_lines.append(f"value: {value[i]:.4f}")
                if show_baseline_text:
                    overlay_lines.append(f"{baseline_label}: {baseline[i]:.4f}")

                img = _draw_overlay_text(
                    img,
                    overlay_lines,
                    xy=(10, 10),
                    fill=(255, 0, 0),
                )

                plot_img = _make_trend_plot(
                    value=value,
                    current_idx=i,
                    baseline=baseline,
                    size=(plot_w, plot_h),
                    title=title,
                    task_title=task_title,
                    baseline_label=baseline_label,
                )

                if plot_img.height != img.height:
                    plot_img = plot_img.resize((plot_img.width, img.height))

                frame = _concat_horizontally(img, plot_img)

                if macro_block_size and macro_block_size > 1:
                    frame = _pad_image_to_multiple(frame, multiple=macro_block_size)

                writer.append_data(np.asarray(frame, dtype=np.uint8))

        shutil.copy(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _process_single_result(args):
    result, eval_type_dir = args

    progress_array = result["progress_pred"]
    task = result["task"]
    sample_id = result["id"]
    video_path = result["video_path"]
    target_progress = result["target_progress"]
    data_source = result["data_source"]

    output_dir = os.path.join(eval_type_dir, data_source)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{sample_id}.mp4")

    if os.path.exists(save_path):
        return save_path

    frames = load_frames_from_npz(video_path)
    images = convert_frames_to_pil_images(frames)

    save_video_with_trend(
        images=images,
        value=progress_array,
        baseline=target_progress,
        output_path=save_path,
        fps=8,
        title="Value Trend",
        task_title=task,
        baseline_label="target_progress",
    )

    return save_path


def _auto_num_workers(num_tasks):
    cpu_count = os.cpu_count() or 32
    # Don't spawn too many workers for video generation; multiprocessing fits
    # this scenario better than multithreading
    return max(1, min(num_tasks, min(cpu_count, 32)))


def save_videos_with_progress(eval_results, eval_type_dir, num_workers=None):
    if not eval_results:
        return

    if num_workers is None:
        num_workers = _auto_num_workers(len(eval_results))

    logger.info(f"Saving {len(eval_results)} videos with {num_workers} processes")

    tasks = [(result, eval_type_dir) for result in eval_results]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_result, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Saving videos"):
            save_path = future.result()
            logger.info(f"Saved {save_path}")
