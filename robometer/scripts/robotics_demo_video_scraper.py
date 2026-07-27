#!/usr/bin/env python3
"""
Scrape robotics company websites for demo videos and download them locally.

Focus: robot manipulation (arms, grasping, assembly, surgical, warehouse picking).
1. Loads (or creates) robotics_companies.txt with company names and URLs (manipulation-focused by default).
2. Visits each URL, finds video links (direct .mp4/.webm, YouTube, Vimeo, video src).
3. Extracts language captions when available (title, aria-label, figcaption, og:title/og:description).
4. Optionally runs a captioning model (e.g. BLIP-2) on sampled frames to describe what the robot is doing.
5. Downloads videos into online_demo_videos/ and saves a companion .txt with instruction, company, url, and model_caption.
6. Keeps a state file (scraped_state.json in out-dir) of processed URLs; reruns skip URLs already downloaded or skipped.

Usage:
  uv run python scripts/robotics_demo_video_scraper.py
  uv run python scripts/robotics_demo_video_scraper.py --companies robotics_companies.txt --out-dir online_demo_videos
  uv run python scripts/robotics_demo_video_scraper.py --caption-model blip2   # add VLM caption per video
  uv run python scripts/robotics_demo_video_scraper.py --crawl-sublinks --max-pages-per-site 20   # follow same-domain links to find more videos
  uv run python scripts/robotics_demo_video_scraper.py --download-youtube   # download YouTube/Vimeo via yt-dlp (pip install yt-dlp)
  uv run python scripts/robotics_demo_video_scraper.py --require-robot --caption-model blip2   # keep only videos where VLM detects a robot

Requires: requests. Captioning: pip install transformers torch (or uv with [robometer]). YouTube/Vimeo download: pip install yt-dlp, then use --download-youtube.
Respect robots.txt and site terms of service; use for personal/educational purposes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests

STATE_FILENAME = "scraped_state.json"
MAX_VIDEO_DURATION_SECONDS = 20.0

OUT_DIR_DEFAULT = Path("online_demo_videos")
COMPANIES_FILE_DEFAULT = Path("robotics_companies.txt")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"

# Starter list: robotics companies and their main / media / news pages
ROBOTICS_COMPANIES = [
    # Humanoid & legged
    ("Boston Dynamics", "https://www.bostondynamics.com"),
    ("Boston Dynamics YouTube", "https://www.youtube.com/@BostonDynamics"),
    ("Agility Robotics", "https://www.agilityrobotics.com"),
    ("Figure", "https://figure.ai"),
    ("1X Technologies", "https://www.1x.tech"),
    ("Unitree Robotics", "https://www.unitree.com"),
    ("Sanctuary AI", "https://www.sanctuary.ai"),
    ("Apptronik", "https://www.apptronik.com"),
    ("Fourier Intelligence", "https://www.fftai.com"),
    ("Tesla Optimus", "https://www.tesla.com"),
    ("Honda Robotics", "https://global.honda/innovation/robotics"),
    ("Hyundai Robotics", "https://www.hyundai-robotics.com"),
    ("Engineered Arts (Ameca)", "https://www.engineeredarts.co.uk"),
    ("PAL Robotics", "https://pal-robotics.com"),
    ("UBTECH", "https://www.ubtrobot.com"),
    ("Flexiv", "https://www.flexiv.com"),
    ("Stretch (Hello Robot)", "https://www.hello-robot.com"),
    # Industrial arms & automation
    ("ABB Robotics", "https://global.abb/group/en/technologies/robotics"),
    ("FANUC", "https://www.fanuc.com"),
    ("KUKA", "https://www.kuka.com"),
    ("Universal Robots", "https://www.universal-robots.com"),
    ("Yaskawa Motoman", "https://www.yaskawa.com"),
    ("Kawasaki Robotics", "https://robotics.kawasaki.com"),
    ("Epson Robots", "https://robots.epson.com"),
    ("DENSO Robotics", "https://www.densorobotics.com"),
    ("Stäubli", "https://www.staubli.com"),
    ("Comau", "https://www.comau.com"),
    ("Nachi Robotics", "https://www.nachirobotics.com"),
    ("Doosan Robotics", "https://www.doosanrobotics.com"),
    ("Techman Robot", "https://www.techmanrobot.com"),
    ("Rethink Robotics (legacy)", "https://www.rethinkrobotics.com"),
    ("Productive Robotics", "https://www.productiverobotics.com"),
    ("Vention", "https://www.vention.io"),
    ("Formic", "https://formic.co"),
    # Medical & surgical
    ("Intuitive (da Vinci)", "https://www.intuitive.com"),
    ("Medtronic (Hugo, Mazor)", "https://www.medtronic.com"),
    ("Stryker (Mako)", "https://www.stryker.com"),
    ("Johnson & Johnson (Ottava)", "https://www.jnj.com"),
    ("CMR Surgical (Versius)", "https://www.cmrsurgical.com"),
    ("Asensus Surgical", "https://www.asensus.com"),
    ("Verb Surgical", "https://www.verbsurgical.com"),
    ("Accuray", "https://www.accuray.com"),
    # Consumer & home
    ("iRobot", "https://www.irobot.com"),
    ("Ecovacs", "https://www.ecovacs.com"),
    ("Roborock", "https://www.roborock.com"),
    ("SharkNinja (robotics)", "https://www.sharkninja.com"),
    ("LG Robotics", "https://www.lg.com"),
    ("Samsung Robotics", "https://www.samsung.com"),
    # Logistics & warehouse
    ("Clearpath Robotics", "https://clearpathrobotics.com"),
    ("Fetch (Zebra)", "https://www.zebra.com/us/en/about-zebra/partners/fetch-robotics.html"),
    ("Locus Robotics", "https://www.locusrobotics.com"),
    ("6 River Systems", "https://6river.com"),
    ("Exotec", "https://www.exotec.com"),
    ("Berkshire Grey", "https://www.berkshiregrey.com"),
    ("AutoStore", "https://www.autostore.com"),
    ("Geek+", "https://www.geekplus.com"),
    ("GreyOrange", "https://www.greyorange.com"),
    ("inVia Robotics", "https://www.inviarobotics.com"),
    ("RightHand Robotics", "https://www.righthandrobotics.com"),
    ("Plus One Robotics", "https://www.plusone.ai"),
    ("Covariant", "https://covariant.ai"),
    # Drones & aerial
    ("DJI", "https://www.dji.com"),
    ("Skydio", "https://www.skydio.com"),
    ("Parrot", "https://www.parrot.com"),
    ("Autel Robotics", "https://www.autelrobotics.com"),
    ("AgEagle", "https://www.ageagle.com"),
    ("Iris Automation", "https://www.irisautomation.com"),
    # Autonomous vehicles & mobility
    ("Waymo", "https://waymo.com"),
    ("Cruise", "https://www.getcruise.com"),
    ("Aurora", "https://aurora.tech"),
    ("Zoox", "https://zoox.com"),
    ("Nuro", "https://www.nuro.ai"),
    ("Starship Technologies", "https://www.starship.xyz"),
    ("KiwiBot", "https://www.kiwicampus.com"),
    ("Ottonomy", "https://www.ottonomy.io"),
    # Research & open source
    ("Open Robotics (ROS)", "https://www.openrobotics.org"),
    ("Willow Garage (legacy)", "https://www.willowgarage.com"),
    ("MIT CSAIL", "https://www.csail.mit.edu"),
    ("CMU Robotics", "https://www.ri.cmu.edu"),
    ("Berkeley AI / RAIL", "https://rail.eecs.berkeley.edu"),
]

# Robot manipulation only: arms, grasping, assembly, surgical, warehouse picking, manipulation research.
# Excludes drones, AVs, consumer vacuums, pure locomotion humanoids.
ROBOTICS_MANIPULATION_COMPANIES = [
    # Industrial arms & automation
    # ("ABB Robotics", "https://global.abb/group/en/technologies/robotics"),
    # ("FANUC", "https://www.fanuc.com"),
    # ("KUKA", "https://www.kuka.com"),
    # ("Universal Robots", "https://www.universal-robots.com"),
    # ("Yaskawa Motoman", "https://www.yaskawa.com"),
    # ("Kawasaki Robotics", "https://robotics.kawasaki.com"),
    # ("Epson Robots", "https://robots.epson.com"),
    # ("DENSO Robotics", "https://www.densorobotics.com"),
    # ("Stäubli", "https://www.staubli.com"),
    # ("Comau", "https://www.comau.com"),
    # ("Nachi Robotics", "https://www.nachirobotics.com"),
    # ("Doosan Robotics", "https://www.doosanrobotics.com"),
    # ("Techman Robot", "https://www.techmanrobot.com"),
    # ("Rethink Robotics (legacy)", "https://www.rethinkrobotics.com"),
    # ("Productive Robotics", "https://www.productiverobotics.com"),
    # ("Vention", "https://www.vention.io"),
    # ("Formic", "https://formic.co"),
    # # Medical & surgical
    # ("Intuitive (da Vinci)", "https://www.intuitive.com"),
    # ("Medtronic (Hugo, Mazor)", "https://www.medtronic.com"),
    # ("Stryker (Mako)", "https://www.stryker.com"),
    # ("Johnson & Johnson (Ottava)", "https://www.jnj.com"),
    # ("CMR Surgical (Versius)", "https://www.cmrsurgical.com"),
    # ("Asensus Surgical", "https://www.asensus.com"),
    # ("Verb Surgical", "https://www.verbsurgical.com"),
    # ("Accuray", "https://www.accuray.com"),
    # # Warehouse picking & manipulation
    # ("RightHand Robotics", "https://www.righthandrobotics.com"),
    # ("Plus One Robotics", "https://www.plusone.ai"),
    # ("Covariant", "https://covariant.ai"),
    # ("Berkshire Grey", "https://www.berkshiregrey.com"),
    # ("inVia Robotics", "https://www.inviarobotics.com"),
    # # Research
    # ("Open Robotics (ROS)", "https://www.openrobotics.org"),
    # ("Willow Garage (legacy)", "https://www.willowgarage.com"),
    # ("MIT CSAIL", "https://www.csail.mit.edu"),
    # ("CMU Robotics", "https://www.ri.cmu.edu"),
    # ("Berkeley AI / RAIL", "https://rail.eecs.berkeley.edu"),
    # # Manipulation-focused arms / humanoids
    # ("Stretch (Hello Robot)", "https://www.hello-robot.com"),
    # ("Flexiv", "https://www.flexiv.com"),
    ("Figure", "https://figure.ai"),
    # ("1X Technologies", "https://www.1x.tech"),
    # ("Apptronik", "https://www.apptronik.com"),
    # ("Sanctuary AI", "https://www.sanctuary.ai"),
]


def ensure_companies_file(path: Path, manipulation_only: bool = True) -> None:
    """Create robotics_companies.txt with default list if it doesn't exist.
    By default uses manipulation-only companies (arms, surgical, picking, research).
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    companies = ROBOTICS_MANIPULATION_COMPANIES if manipulation_only else ROBOTICS_COMPANIES
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Robot manipulation companies: one per line as 'Name<TAB>URL' or 'Name, URL'\n")
        for name, url in companies:
            f.write(f"{name}\t{url}\n")
    print(f"Wrote {path} with {len(companies)} entries (manipulation_only={manipulation_only}). Edit to add more.")


def load_companies(path: Path) -> list[tuple[str, str]]:
    """Load (name, url) pairs from file. Lines: 'Name\\tURL' or 'Name, URL'; # ignored."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                name, url = line.split("\t", 1)
            else:
                name, _, url = line.partition(",")
                name, url = name.strip(), url.strip()
            if name and url and url.startswith("http"):
                pairs.append((name, url))
    return pairs


def _extract_page_caption(html: str) -> str:
    """Extract page-level caption from og:title and og:description."""
    parts: list[str] = []
    for prop, name in [("og:title", "title"), ("og:description", "description")]:
        m = re.search(
            rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.I,
        )
        if not m:
            m = re.search(
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(prop)}["\']',
                html,
                re.I,
            )
        if m:
            parts.append(m.group(1).strip())
    return " | ".join(parts)[:2000] if parts else ""


def _extract_caption_near(html: str, start: int, end: int) -> str:
    """Extract caption from HTML segment (title=, aria-label=, data-caption=, figcaption)."""
    segment = html[max(0, start - 600) : end + 200]
    # title="..." or aria-label="..."
    for attr in ("title", "aria-label", "data-caption", "data-title", "data-description"):
        m = re.search(rf'{attr}=["\']([^"\']+)["\']', segment, re.I)
        if m:
            s = m.group(1).strip()
            if len(s) > 10 and "script" not in s.lower():
                return re.sub(r"\s+", " ", s)[:2000]
    # <figcaption>...</figcaption>
    m = re.search(r"<figcaption[^>]*>([^<]+(?:<[^>]+>[^<]*)*)</figcaption>", segment, re.I | re.DOTALL)
    if m:
        text = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
        text = re.sub(r"\s+", " ", text)[:2000]
        if len(text) > 5:
            return text
    return ""


def find_video_urls_in_html(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract video URLs and optional captions from HTML. Returns list of (url, caption)."""
    base = base_url.rstrip("/")
    page_caption = _extract_page_caption(html)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str, caption: str = "") -> None:
        if url not in seen and url.startswith("http"):
            seen.add(url)
            cap = caption.strip() or page_caption
            results.append((url, cap))

    # Direct video extensions (with optional caption from surrounding HTML)
    ext_pat = re.compile(
        r"\b(href|src|content)=[\"']([^\"']+?\.(?:mp4|webm|mov|m4v|ogv))[\"']",
        re.I,
    )
    for m in re.finditer(ext_pat, html):
        url = urljoin(base + "/", m.group(2))
        cap = _extract_caption_near(html, m.start(), m.end())
        add(url, cap)

    # Any href or src with video-like path
    url_in_attr = re.compile(
        r'(?:href|src|content)=["\']([^"\']+)["\']',
        re.I,
    )
    for m in re.finditer(url_in_attr, html):
        raw = m.group(1).strip()
        if not raw or raw.startswith("#") or raw.startswith("javascript:"):
            continue
        full = urljoin(base + "/", raw)
        if full.startswith("http") and (
            ".mp4" in full
            or ".webm" in full
            or ".mov" in full
            or "youtube.com" in full
            or "youtu.be" in full
            or "vimeo.com" in full
            or "video" in full.lower()
            or "/v/" in full
        ):
            cap = _extract_caption_near(html, m.start(), m.end())
            add(full, cap)

    # YouTube embed (caption from iframe title if present)
    yt = re.compile(
        r"(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([a-zA-Z0-9_-]{11})"
    )
    for m in re.finditer(yt, html):
        url = f"https://www.youtube.com/watch?v={m.group(1)}"
        cap = _extract_caption_near(html, m.start(), m.end())
        add(url, cap)

    # Vimeo
    vimeo = re.compile(r"vimeo\.com/(?:video/)?(\d+)")
    for m in re.finditer(vimeo, html):
        url = f"https://vimeo.com/{m.group(1)}"
        cap = _extract_caption_near(html, m.start(), m.end())
        add(url, cap)

    return results


def _same_domain_links(html: str, base_url: str, netloc: str) -> set[str]:
    """Extract href URLs that belong to the same domain as base_url. Returns absolute URLs."""
    base = base_url.rstrip("/")
    seen: set[str] = set()
    # href="..."
    for m in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', html, re.I):
        raw = m.group(1).strip()
        if not raw or raw.startswith("#") or raw.startswith("javascript:") or raw.startswith("mailto:"):
            continue
        full = urljoin(base + "/", raw)
        parsed = urlparse(full)
        if parsed.netloc != netloc or not full.startswith("http"):
            continue
        # Skip obvious non-HTML (files we don't want to fetch as pages)
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in (".pdf", ".zip", ".mp4", ".webm", ".mov", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js")):
            continue
        seen.add(full)
    return seen


def _crawl_site_for_videos(
    start_url: str,
    name: str,
    session: requests.Session,
    delay: float,
    max_pages: int,
) -> list[tuple[str, str]]:
    """Crawl same-domain pages starting from start_url and collect all (video_url, caption) pairs."""
    parsed_start = urlparse(start_url)
    netloc = parsed_start.netloc
    results: list[tuple[str, str]] = []
    seen_pages: set[str] = set()
    queue: list[str] = [start_url.rstrip("/")]
    pages_fetched = 0

    while queue and pages_fetched < max_pages:
        url = queue.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)
        pages_fetched += 1
        try:
            r = session.get(url, timeout=15)
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print(f"  Skip subpage {url[:60]}...: {e}")
            time.sleep(delay)
            continue
        for u, caption in find_video_urls_in_html(html, url):
            results.append((u, caption))
        # Enqueue same-domain links we haven't visited
        for link in _same_domain_links(html, url, netloc):
            if link not in seen_pages and link not in queue:
                queue.append(link)
        time.sleep(delay)

    return results


def _get_video_duration_seconds(path: Path) -> float | None:
    """Return video duration in seconds via ffprobe, or None if unknown/unavailable."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return float(out.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return None


def _is_mp4_file(path: Path) -> bool:
    """Return True if file looks like an MP4 (ftyp at offset 4)."""
    try:
        with open(path, "rb") as f:
            f.seek(4)
            return f.read(4) == b"ftyp"
    except Exception:
        return False


def _is_likely_html(path: Path) -> bool:
    """Return True if file looks like HTML (common when server returns error page)."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
        return head.lstrip().startswith((b"<", b"\r\n", b"\n")) or b"<!DOCTYPE" in head[:200]
    except Exception:
        return False


def _convert_to_mp4(path: Path) -> bool:
    """Convert video to MP4 with ffmpeg (remux or re-encode). Returns True if successful."""
    if _is_mp4_file(path):
        return True
    tmp = path.with_suffix(".tmp.mp4")
    try:
        # Try remux first (fast); then re-encode if needed
        out = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-c", "copy", "-movflags", "+faststart",
                str(tmp),
            ],
            capture_output=True,
            timeout=120,
        )
        if out.returncode != 0:
            out = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
                    "-movflags", "+faststart", str(tmp),
                ],
                capture_output=True,
                timeout=300,
            )
        if out.returncode != 0:
            return False
        path.unlink(missing_ok=True)
        tmp.rename(path)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        tmp.unlink(missing_ok=True)
        return False
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def download_direct(url: str, path: Path, session: requests.Session) -> bool:
    """Download a direct video URL and ensure output is valid .mp4. Returns True if successful.
    Caller should pass path with .mp4 extension; non-MP4 content is converted via ffmpeg.
    """
    try:
        r = session.get(url, stream=True, timeout=30)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"  Skip {url[:60]}...: {e}")
        return False

    # Reject HTML (error pages)
    if _is_likely_html(path):
        path.unlink(missing_ok=True)
        print(f"  Skip (not video, got HTML): {path.name}")
        return False

    # Ensure valid MP4: convert with ffmpeg if not already MP4
    if not _is_mp4_file(path):
        if not _convert_to_mp4(path):
            path.unlink(missing_ok=True)
            print(f"  Skip (ffmpeg convert failed): {path.name}")
            return False
    return True


def _youtube_vimeo_id(url: str) -> str | None:
    """Extract video id from YouTube or Vimeo URL for stable filenames. Returns None if not recognized."""
    if "youtube.com" in url or "youtu.be" in url:
        if "youtu.be/" in url:
            m = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
            return m.group(1) if m else None
        parsed = urlparse(url)
        if parsed.netloc and "youtube" in parsed.netloc:
            q = parse_qs(parsed.query)
            v = q.get("v", [])
            return v[0] if v else None
    if "vimeo.com" in url:
        m = re.search(r"vimeo\.com/(?:video/)?(\d+)", url)
        return m.group(1) if m else None
    return None


def download_youtube_or_vimeo(url: str, output_path: Path) -> Path | None:
    """Download a YouTube or Vimeo URL with yt-dlp; output is converted to .mp4. Returns final path or None."""
    if not shutil.which("yt-dlp"):
        print("  yt-dlp not found; install with: pip install yt-dlp")
        return None
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # yt-dlp writes to stem.%(ext)s; we'll get stem.mkv or stem.webm etc., then convert to stem.mp4
    stem = output_path.with_suffix("")
    out_tpl = str(stem) + ".%(ext)s"
    try:
        out = subprocess.run(
            [
                "yt-dlp",
                "--no-warnings",
                "-o",
                out_tpl,
                "--no-playlist",
                "--max-downloads", "1",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode != 0:
            print(f"  yt-dlp failed: {out.stderr[:200] if out.stderr else out.stdout[:200]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"  yt-dlp error: {e}")
        return None
    # Find the file yt-dlp wrote (stem with some extension)
    downloaded: Path | None = None
    for f in output_path.parent.iterdir():
        if f.stem == stem.name and f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov", ".m4a"):
            downloaded = f
            break
    if not downloaded or not downloaded.is_file():
        return None
    # Ensure final file is .mp4
    final = stem.with_suffix(".mp4")
    if downloaded.suffix.lower() != ".mp4" or not _is_mp4_file(downloaded):
        if not _convert_to_mp4(downloaded):
            downloaded.unlink(missing_ok=True)
            return None
        if downloaded.suffix.lower() != ".mp4":
            downloaded.rename(final)
        else:
            final = downloaded
    else:
        if downloaded != final:
            downloaded.rename(final)
    return final if final.exists() else None


def _sample_frames(video_path: Path, num_frames: int = 3) -> list:
    """Sample num_frames from video (start, middle, end). Returns list of PIL Images."""
    try:
        from decord import VideoReader  # type: ignore
    except ImportError:
        raise RuntimeError("decord is required for captioning; pip install decord")
    import numpy as np
    from PIL import Image

    vr = VideoReader(str(video_path), num_threads=1)
    n = len(vr)
    if n == 0:
        return []
    indices = [
        int(i * (n - 1) / max(1, num_frames - 1))
        for i in range(num_frames)
    ]
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
    return [Image.fromarray(f).convert("RGB") for f in frames]


# Lazy-loaded caption model (processor, model) keyed by model_id
_caption_model_cache: dict[str, tuple] = {}


def _caption_video_with_model(video_path: Path, model_id: str, prompt: str = "What is the robot doing in this image?") -> str:
    """Run a vision-language model on sampled frames and return a single caption for the video.
    model_id: e.g. 'blip2' (Salesforce/blip2-opt-2.7b) or any HuggingFace model id for Blip2ForConditionalGeneration.
    """
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError("transformers and torch required for captioning; pip install transformers torch") from e

    if model_id.lower() == "blip2":
        model_id = "Salesforce/blip2-opt-2.7b"

    if model_id not in _caption_model_cache:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _caption_model_cache[model_id] = (processor, model, device)

    processor, model, device = _caption_model_cache[model_id]
    images = _sample_frames(video_path, num_frames=3)
    if not images:
        return ""

    captions: list[str] = []
    for img in images:
        inputs = processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=80)
        cap = processor.decode(out[0], skip_special_tokens=True).strip()
        if cap and cap != prompt:
            captions.append(cap)
    if not captions:
        return ""
    # Prefer a single representative caption; if we have several, take the longest (often most descriptive)
    return max(captions, key=len)


def _query_vlm_yes_no(video_path: Path, model_id: str, question: str) -> bool:
    """Run VLM on sampled frames with a yes/no question; return True if answer suggests yes.
    Uses same BLIP-2 pipeline as captioning. Parses first few tokens for yes/no/robot.
    """
    raw = _caption_video_with_model(video_path, model_id, prompt=question)
    if not raw:
        return False
    s = raw.strip().lower()
    if s.startswith("yes") or s.startswith("yeah"):
        return True
    if s.startswith("no") or s.startswith("nope"):
        return False
    # Answer describes something; treat as yes if it mentions a robot (and doesn't negate)
    if "robot" in s and not s.startswith("no ") and "no robot" not in s[:30]:
        return True
    return False


def _video_has_robot(video_path: Path, model_id: str) -> bool:
    """Return True if VLM indicates a robot is present in the video (sample frames)."""
    question = "Is there a robot in this image? Answer yes or no."
    return _query_vlm_yes_no(video_path, model_id, question)


def _caption_to_instruction(caption: str, company: str) -> str:
    """Turn scraped caption into a language instruction for the video (task description)."""
    if caption and len(caption.strip()) > 5:
        # Use as-is or normalize to imperative/task form
        s = caption.strip()
        s = re.sub(r"\s+", " ", s)
        # If it looks like a title (no verb), prefix with "Demonstrate: " or use as task
        if len(s) < 200 and not any(
            s.strip().lower().startswith(p)
            for p in ("the robot", "robot", "a ", "demonstrate", "show", "perform")
        ):
            return f"Demonstrate: {s}" if not s.endswith(".") else s
        return s[:2000]
    if company:
        return f"Robot demonstration from {company}."
    return "Robot demonstration."


def save_caption(
    path: Path,
    caption: str,
    company: str,
    url: str,
    model_caption: str | None = None,
) -> None:
    """Write companion .txt with instruction (language task for the video), company, url, and optional model_caption."""
    # Prefer model-generated caption for instruction when available
    if model_caption and len(model_caption.strip()) > 5:
        instruction = model_caption.strip()[:2000]
        if not instruction.endswith("."):
            instruction += "."
    else:
        instruction = _caption_to_instruction(caption, company)
    txt_path = path.with_suffix(".txt")
    lines = [
        "instruction: " + instruction,
        "",
        "company: " + (company or ""),
        "url: " + (url or ""),
    ]
    if model_caption and model_caption.strip():
        lines.append("")
        lines.append("model_caption: " + model_caption.strip()[:2000])
    if caption and caption.strip() != instruction:
        lines.append("")
        lines.append("caption: " + caption.strip()[:2000])
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_filename(name: str, max_len: int = 80) -> str:
    """Make a safe filename from a string."""
    s = re.sub(r"[^\w\s\-\.]", "", name)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] or "video"


def _state_path(out_dir: Path) -> Path:
    return out_dir / STATE_FILENAME


def load_state(out_dir: Path) -> dict[str, str]:
    """Load processed URL -> status from scraped_state.json. Status: downloaded, skipped_robot, skipped_fail, skipped_too_long."""
    path = _state_path(out_dir)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("by_url", data) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(out_dir: Path, state: dict[str, str]) -> None:
    """Write processed URL -> status to scraped_state.json."""
    path = _state_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"by_url": state, "version": 1}, f, indent=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape robotics company sites for demo videos and download them."
    )
    parser.add_argument(
        "--companies",
        type=Path,
        default=COMPANIES_FILE_DEFAULT,
        help="Path to .txt with company names and URLs (Name\\tURL per line)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR_DEFAULT,
        help="Directory to save videos",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between requests to the same site",
    )
    parser.add_argument(
        "--skip-youtube-vimeo",
        action="store_true",
        help="Do not list YouTube/Vimeo (only download direct links unless yt-dlp used)",
    )
    parser.add_argument(
        "--all-companies",
        action="store_true",
        help="Use full robotics list (drones, AVs, etc.) when creating companies file; default is manipulation-only",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default=None,
        metavar="ID",
        help="Run a vision model to caption each video (e.g. blip2 or Salesforce/blip2-opt-2.7b). Requires transformers, torch, decord.",
    )
    parser.add_argument(
        "--crawl-sublinks",
        action="store_true",
        help="Follow same-domain links on each site to find videos on subpages (not just the main URL).",
    )
    parser.add_argument(
        "--max-pages-per-site",
        type=int,
        default=20,
        metavar="N",
        help="When using --crawl-sublinks, limit to N pages per company site (default 20).",
    )
    parser.add_argument(
        "--download-youtube",
        action="store_true",
        help="Download YouTube/Vimeo videos via yt-dlp into out-dir (pip install yt-dlp).",
    )
    parser.add_argument(
        "--require-robot",
        action="store_true",
        help="After download, run VLM to check if a robot is in the video; keep only if yes (uses --caption-model or blip2).",
    )
    args = parser.parse_args()

    ensure_companies_file(args.companies, manipulation_only=not args.all_companies)
    companies = load_companies(args.companies)
    if not companies:
        print("No companies found in", args.companies)
        return

    print(f"Loaded {len(companies)} companies. Output dir: {args.out_dir.absolute()}")
    if args.crawl_sublinks:
        print(f"  Crawl sublinks: up to {args.max_pages_per_site} pages per site")
    if args.download_youtube:
        print("  Download YouTube/Vimeo: yes (yt-dlp)")
    if args.require_robot:
        print("  Require robot in video: yes (VLM filter)")
    if args.caption_model:
        print(f"  Caption model: {args.caption_model}")
    print(f"  Max video duration: {MAX_VIDEO_DURATION_SECONDS:.0f}s (longer videos skipped)")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(args.out_dir)
    if state:
        print(f"Loaded state: {len(state)} URL(s) already processed (will skip)")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    all_direct: list[tuple[str, str, str, str]] = []  # (company_name, url, suggested_path, caption)
    yt_vimeo: list[tuple[str, str, str]] = []  # (company_name, url, caption)

    for name, url in companies:
        if args.crawl_sublinks:
            print(f"Crawling ({args.max_pages_per_site} pages max): {name} — {url}")
            try:
                page_results = _crawl_site_for_videos(
                    url, name, session, args.delay, args.max_pages_per_site
                )
            except Exception as e:
                print(f"  Error: {e}")
                continue
        else:
            print(f"Fetching: {name} — {url}")
            try:
                r = session.get(url, timeout=15)
                r.raise_for_status()
                html = r.text
            except Exception as e:
                print(f"  Error: {e}")
                continue
            page_results = find_video_urls_in_html(html, url)
            time.sleep(args.delay)
        n_direct = 0
        n_yt = 0
        for u, caption in page_results:
            if "youtube.com" in u or "youtu.be" in u or "vimeo.com" in u:
                yt_vimeo.append((name, u, caption))
                n_yt += 1
            else:
                slug = safe_filename(name)
                url_hash = str(abs(hash(u)) % 10**8)[:8]
                all_direct.append((name, u, f"{slug}_{url_hash}.mp4", caption))
                n_direct += 1
        print(f"  Found {n_direct} direct video(s), {n_yt} YouTube/Vimeo link(s)")
        if args.crawl_sublinks:
            time.sleep(args.delay)

    # Deduplicate by URL for direct (keep first occurrence and its caption)
    n_unique_direct = len({u for _, u, _, _ in all_direct})
    print(f"\nTotal: {len(all_direct)} direct video URL(s) (before dedup), {n_unique_direct} unique; {len(yt_vimeo)} YouTube/Vimeo URL(s)")
    robot_check_model: str | None = (args.caption_model or "blip2") if args.require_robot else None
    if robot_check_model:
        print(f"Robot check enabled (model: {robot_check_model})")
    print("\n--- Direct videos ---")
    seen = set()
    n_direct_kept = 0
    n_direct_skipped_robot = 0
    n_direct_skipped_fail = 0
    n_direct_skipped_too_long = 0
    n_direct_existed = 0
    n_direct_skipped_state = 0
    for name, u, fname, caption in all_direct:
        if u in seen:
            continue
        seen.add(u)
        if u in state:
            n_direct_skipped_state += 1
            continue
        path = args.out_dir / fname
        model_caption: str | None = None
        if args.caption_model and path.exists():
            try:
                print(f"Captioning: {path.name}")
                model_caption = _caption_video_with_model(path, args.caption_model)
            except Exception as e:
                print(f"  Caption error: {e}")
        if path.exists():
            print(f"Exists: {path.name}")
            save_caption(path, caption, name, u, model_caption=model_caption)
            state[u] = "downloaded"
            save_state(args.out_dir, state)
            n_direct_existed += 1
            time.sleep(args.delay)
            continue
        print(f"Downloading: {path.name}")
        if download_direct(u, path, session):
            duration = _get_video_duration_seconds(path)
            if duration is not None and duration > MAX_VIDEO_DURATION_SECONDS:
                path.unlink(missing_ok=True)
                print(f"  Skip (video > {MAX_VIDEO_DURATION_SECONDS:.0f}s): {path.name} ({duration:.1f}s)")
                state[u] = "skipped_too_long"
                save_state(args.out_dir, state)
                n_direct_skipped_too_long += 1
                time.sleep(args.delay)
                continue
            if robot_check_model:
                try:
                    print(f"  Checking for robot: {path.name}")
                    if not _video_has_robot(path, robot_check_model):
                        path.unlink(missing_ok=True)
                        print(f"  Skip (no robot detected): {path.name}")
                        state[u] = "skipped_robot"
                        save_state(args.out_dir, state)
                        n_direct_skipped_robot += 1
                        time.sleep(args.delay)
                        continue
                    print(f"  Robot detected, keeping.")
                except Exception as e:
                    print(f"  Robot check error: {e}")
            if args.caption_model:
                try:
                    print(f"  Captioning: {path.name}")
                    model_caption = _caption_video_with_model(path, args.caption_model)
                except Exception as e:
                    print(f"  Caption error: {e}")
            save_caption(path, caption, name, u, model_caption=model_caption)
            print(f"  Saved: {path.name} (+ .txt)")
            state[u] = "downloaded"
            save_state(args.out_dir, state)
            n_direct_kept += 1
        else:
            state[u] = "skipped_fail"
            save_state(args.out_dir, state)
            n_direct_skipped_fail += 1
        time.sleep(args.delay)

    # YouTube/Vimeo: optionally download with yt-dlp, or just list
    if yt_vimeo:
        if args.download_youtube:
            n_yt_unique = len({u for _, u, _ in yt_vimeo})
            print(f"\n--- YouTube/Vimeo ({n_yt_unique} unique) ---")
            seen_yt = set()
            n_yt_kept = 0
            n_yt_skipped_robot = 0
            n_yt_skipped_too_long = 0
            n_yt_existed = 0
            n_yt_skipped_state = 0
            for name, u, caption in yt_vimeo:
                if u in seen_yt:
                    continue
                seen_yt.add(u)
                if u in state:
                    n_yt_skipped_state += 1
                    continue
                vid = _youtube_vimeo_id(u)
                fname = f"{safe_filename(name)}_{vid or abs(hash(u)) % 10**8}.mp4"
                path = args.out_dir / fname
                if path.exists():
                    print(f"Exists (YT/Vimeo): {path.name}")
                    save_caption(path, caption, name, u)
                    state[u] = "downloaded"
                    save_state(args.out_dir, state)
                    n_yt_existed += 1
                    time.sleep(args.delay)
                    continue
                print(f"Downloading (yt-dlp): {path.name}")
                final = download_youtube_or_vimeo(u, path)
                if final:
                    print(f"  Downloaded: {final.name}")
                    duration = _get_video_duration_seconds(final)
                    if duration is not None and duration > MAX_VIDEO_DURATION_SECONDS:
                        final.unlink(missing_ok=True)
                        print(f"  Skip (video > {MAX_VIDEO_DURATION_SECONDS:.0f}s): {final.name} ({duration:.1f}s)")
                        state[u] = "skipped_too_long"
                        save_state(args.out_dir, state)
                        n_yt_skipped_too_long += 1
                        time.sleep(args.delay)
                        continue
                    if robot_check_model:
                        try:
                            print(f"  Checking for robot: {final.name}")
                            if not _video_has_robot(final, robot_check_model):
                                final.unlink(missing_ok=True)
                                print(f"  Skip (no robot detected): {final.name}")
                                state[u] = "skipped_robot"
                                save_state(args.out_dir, state)
                                n_yt_skipped_robot += 1
                                time.sleep(args.delay)
                                continue
                            print(f"  Robot detected, keeping.")
                        except Exception as e:
                            print(f"  Robot check error: {e}")
                    model_caption = None
                    if args.caption_model:
                        try:
                            print(f"  Captioning: {final.name}")
                            model_caption = _caption_video_with_model(final, args.caption_model)
                        except Exception as e:
                            print(f"  Caption error: {e}")
                    save_caption(final, caption, name, u, model_caption=model_caption)
                    print(f"  Saved: {final.name} (+ .txt)")
                    state[u] = "downloaded"
                    save_state(args.out_dir, state)
                    n_yt_kept += 1
                else:
                    state[u] = "skipped_fail"
                    save_state(args.out_dir, state)
                time.sleep(args.delay)
        elif not args.skip_youtube_vimeo:
            print("\nYouTube/Vimeo URLs (use --download-youtube to download via yt-dlp):")
            for name, u, cap in yt_vimeo[:30]:
                print(f"  {name}: {u}")
                if cap:
                    print(f"    caption: {(cap[:80] + '...') if len(cap) > 80 else cap}")
            if len(yt_vimeo) > 30:
                print(f"  ... and {len(yt_vimeo) - 30} more")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Direct: {n_direct_kept} new, {n_direct_existed} already existed, {n_direct_skipped_state} skipped (in state), {n_direct_skipped_too_long} skipped (>{MAX_VIDEO_DURATION_SECONDS:.0f}s), {n_direct_skipped_robot} skipped (no robot), {n_direct_skipped_fail} download failed")
    if yt_vimeo and args.download_youtube:
        print(f"  YouTube/Vimeo: {n_yt_kept} new, {n_yt_existed} already existed, {n_yt_skipped_state} skipped (in state), {n_yt_skipped_too_long} skipped (>{MAX_VIDEO_DURATION_SECONDS:.0f}s), {n_yt_skipped_robot} skipped (no robot)")
    print(f"  Output: {args.out_dir.absolute()}")
    print("Done.")


if __name__ == "__main__":
    main()
