import io
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


def _fig_to_pil(fig, size=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    plt.close(fig)
    if size is not None:
        img = img.resize(size)
    return img


def _format_time(seconds):
    minutes = int(seconds // 60)
    seconds_int = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{seconds_int:02d}.{millis:03d}"


def _make_remaining_time_curve(num_frames, fps):
    """
    Remaining time (seconds) from frame i to the last frame
    """
    if num_frames <= 0:
        return np.array([], dtype=float)
    indices = np.arange(num_frames)
    return (num_frames - 1 - indices) / float(fps)


def _make_trend_plot(
    value,
    current_idx,
    fps,
    size=(400, 300),
    title="Value Trend",
    task_title=None,
    baseline_label="remaining time",
    sampled_indices=None,
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

    fig, ax1 = plt.subplots(
        figsize=(fig_w, fig_h),
        dpi=dpi,
        constrained_layout=True
    )

    if sampled_indices is not None:
        x = np.asarray(sampled_indices, dtype=float)
        total_frames = int(x[-1]) + 1
        remaining_curve = (x[-1] - x) / float(fps)
    else:
        x = np.arange(len(value))
        total_frames = len(value)
        remaining_curve = _make_remaining_time_curve(len(value), fps)
    y = np.asarray(value, dtype=float)

    ax1.plot(
        x[: current_idx + 1],
        y[: current_idx + 1],
        color="tab:blue",
        linewidth=line_w,
        label="value"
    )
    ax1.scatter(
        [x[current_idx]],
        [y[current_idx]],
        color="red",
        s=marker_s1,
        zorder=3,
        label="current value"
    )
    ax1.set_xlabel("Frame", fontsize=label_fs)
    ax1.set_ylabel("Value", color="tab:blue", fontsize=label_fs)
    ax1.tick_params(axis="x", labelsize=tick_fs)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=tick_fs)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(total_frames - 1, 1))

    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    margin = 0.05 * (y_max - y_min)
    ax1.set_ylim(y_min - margin, y_max + margin)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        remaining_curve,
        color="green",
        linestyle="--",
        linewidth=line_w,
        label=baseline_label
    )
    ax2.scatter(
        [x[current_idx]],
        [remaining_curve[current_idx]],
        color="green",
        s=marker_s2,
        zorder=3,
        label="current remaining"
    )

    right_ylabel = "Remain (s)" if small_mode else "Remaining Time (s)"
    ax2.set_ylabel(right_ylabel, color="green", fontsize=label_fs)
    ax2.tick_params(axis="y", labelcolor="green", labelsize=tick_fs)

    rt_min, rt_max = float(np.min(remaining_curve)), float(np.max(remaining_curve))
    if rt_min == rt_max:
        rt_min -= 1.0
        rt_max += 1.0
    rt_margin = 0.05 * (rt_max - rt_min)
    ax2.set_ylim(rt_min - rt_margin, rt_max + rt_margin)

    if small_mode:
        full_title = title
    else:
        full_title = title if task_title is None else f"{task_title}\n{title}"
    ax1.set_title(full_title, fontsize=title_fs)

    if show_legend:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
            fontsize=legend_fs
        )

    return _fig_to_pil(fig, size=size)


def _concat_horizontally(img1, img2, bg_color=(255, 255, 255)):
    h = max(img1.height, img2.height)
    w = img1.width + img2.width
    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    return canvas


def _draw_overlay_text(img, lines, xy=(10, 10), fill=(255, 0, 0), line_spacing=6):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

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
    output_path,
    fps=30,
    plot_width_ratio=0.45,
    title="Value Trend",
    task_title=None,
    baseline_label="remaining time",
    show_remaining_time_text=True,
    show_value_text=True,
    show_task_title=True,
    min_plot_width=260,
    max_plot_width=520,
    min_plot_height=220,
    codec="libx264",
    macro_block_size=16,
    sampled_indices=None,
):
    if len(images) == 0:
        raise ValueError("images is empty")
    if sampled_indices is not None:
        if len(value) != len(sampled_indices):
            raise ValueError(
                f"len(value)={len(value)} != len(sampled_indices)={len(sampled_indices)}"
            )
    else:
        if len(images) != len(value):
            raise ValueError(f"len(images)={len(images)} != len(value)={len(value)}")

    images = [img.convert("RGB") for img in images]
    base_w, base_h = images[0].size

    if base_h < min_plot_height:
        scale = float(min_plot_height) / float(base_h)
        base_w = int(round(base_w * scale))
        base_h = min_plot_height

    images = [img.resize((base_w, base_h)) for img in images]

    plot_w = int(base_w * plot_width_ratio)
    plot_w = max(min_plot_width, min(plot_w, max_plot_width))
    plot_h = max(min_plot_height, base_h)

    num_frames = len(images)
    remaining_curve = _make_remaining_time_curve(num_frames, fps)

    if sampled_indices is not None:
        sampled_set = set(sampled_indices)
        idx_to_value_pos = {idx: pos for pos, idx in enumerate(sampled_indices)}
    else:
        sampled_set = None
        idx_to_value_pos = None

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec=codec,
        macro_block_size=macro_block_size,
    )

    plot_current_idx = 0
    for i, img in enumerate(images):
        overlay_lines = []

        if show_task_title and task_title is not None:
            overlay_lines.append(f"task: {task_title}")

        if sampled_indices is not None:
            if i in sampled_set:
                plot_current_idx = idx_to_value_pos[i]
            if show_value_text:
                overlay_lines.append(f"value: {value[plot_current_idx]:.4f}")
        else:
            plot_current_idx = i
            if show_value_text:
                overlay_lines.append(f"value: {value[i]:.4f}")

        if show_remaining_time_text:
            overlay_lines.append(f"remaining: {_format_time(remaining_curve[i])}")

        img = _draw_overlay_text(img, overlay_lines, xy=(10, 10), fill=(255, 0, 0))

        plot_img = _make_trend_plot(
            value=value,
            current_idx=plot_current_idx,
            fps=fps,
            size=(plot_w, plot_h),
            title=title,
            task_title=task_title,
            baseline_label=baseline_label,
            sampled_indices=sampled_indices,
        )

        if plot_img.height != img.height:
            plot_img = plot_img.resize((plot_img.width, img.height))

        frame = _concat_horizontally(img, plot_img)

        # Proactively pad to a multiple of macro_block_size to avoid ffmpeg warnings
        if macro_block_size and macro_block_size > 1:
            frame = _pad_image_to_multiple(frame, multiple=macro_block_size)

        writer.append_data(np.array(frame))

    writer.close()
