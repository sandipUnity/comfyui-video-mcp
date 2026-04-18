"""
Video montage compiler — concatenates scene videos into a single final video.

Two backends, tried in order:
  1. moviepy  — if installed (pip install moviepy)
                Supports crossfade/dissolve transitions between clips.
  2. ffmpeg   — subprocess call (ffmpeg must be on PATH)
                Uses concat demuxer for simple cuts (fastest, no quality loss).

If neither is available, raises RuntimeError with install instructions.

Usage:
    from pipeline.montage import compile_montage, has_montage_support

    if has_montage_support():
        output = compile_montage(
            video_paths=[Path("s1.mp4"), Path("s2.mp4"), Path("s3.mp4")],
            output_path=Path("output/final.mp4"),
            transition="dissolve",
            transition_duration=0.5,
            music_path=None,
            music_volume=0.3,
            fps=25,
        )
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ── Backend detection ─────────────────────────────────────────────────────────

def _has_moviepy() -> bool:
    try:
        import moviepy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def has_montage_support() -> bool:
    """Return True if at least one compilation backend is available."""
    return _has_moviepy() or _has_ffmpeg()


def available_backend() -> str:
    """Return 'moviepy', 'ffmpeg', or 'none'."""
    if _has_moviepy():
        return "moviepy"
    if _has_ffmpeg():
        return "ffmpeg"
    return "none"


# ── Public API ────────────────────────────────────────────────────────────────

def compile_montage(
    video_paths: list[Path],
    output_path: Path,
    transition: str = "dissolve",   # "dissolve" | "fade" | "cut"
    transition_duration: float = 0.5,
    music_path: Optional[Path] = None,
    music_volume: float = 0.3,
    fps: int = 25,
) -> Path:
    """Compile a list of video files into one final video.

    Args:
        video_paths:         Ordered list of scene video files.
        output_path:         Destination for the compiled video.
        transition:          "dissolve" (crossfade), "fade" (fade through black), "cut".
        transition_duration: Overlap duration in seconds (dissolve/fade only).
        music_path:          Optional background music file. Mixed at *music_volume*.
        music_volume:        Background music volume 0.0-1.0.
        fps:                 Output frame rate.

    Returns:
        Path to the compiled output file.

    Raises:
        RuntimeError: No backend available, or compilation fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_paths = [Path(p) for p in video_paths if Path(p).exists()]
    if not video_paths:
        raise RuntimeError("No valid video files to compile.")
    if len(video_paths) == 1:
        # Single clip — just copy
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return output_path

    backend = available_backend()
    if backend == "moviepy":
        return _compile_moviepy(video_paths, output_path, transition,
                                transition_duration, music_path, music_volume, fps)
    elif backend == "ffmpeg":
        return _compile_ffmpeg(video_paths, output_path, transition,
                               transition_duration, music_path, music_volume, fps)
    else:
        raise RuntimeError(
            "No video compilation backend found.\n\n"
            "Install one of:\n"
            "  pip install moviepy\n"
            "  — or —\n"
            "  Download ffmpeg from https://ffmpeg.org/download.html and add to PATH"
        )


# ── moviepy backend ───────────────────────────────────────────────────────────

def _compile_moviepy(
    paths: list[Path],
    output: Path,
    transition: str,
    t_dur: float,
    music_path: Optional[Path],
    music_vol: float,
    fps: int,
) -> Path:
    from moviepy.editor import (
        VideoFileClip,
        concatenate_videoclips,
        AudioFileClip,
        CompositeAudioClip,
    )

    clips = [VideoFileClip(str(p)) for p in paths]

    if transition in ("dissolve", "fade") and t_dur > 0:
        method = "compose" if transition == "dissolve" else "compose"
        # Apply crossfade transitions between clips
        for i in range(1, len(clips)):
            clips[i] = clips[i].crossfadein(t_dur)
        final = concatenate_videoclips(clips, method="compose",
                                       padding=-t_dur if transition == "dissolve" else 0)
    else:
        # Simple cut
        final = concatenate_videoclips(clips, method="chain")

    # Add background music if provided
    if music_path and Path(music_path).exists():
        bg = (
            AudioFileClip(str(music_path))
            .subclip(0, min(final.duration, AudioFileClip(str(music_path)).duration))
            .volumex(music_vol)
        )
        if final.audio:
            final = final.set_audio(CompositeAudioClip([final.audio, bg]))
        else:
            final = final.set_audio(bg)

    final.write_videofile(
        str(output),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    for c in clips:
        c.close()

    return output


# ── ffmpeg backend ────────────────────────────────────────────────────────────

def _compile_ffmpeg(
    paths: list[Path],
    output: Path,
    transition: str,
    t_dur: float,
    music_path: Optional[Path],
    music_vol: float,
    fps: int,
) -> Path:
    """FFmpeg backend. Uses concat demuxer (fast, lossless). Transitions via xfade filter."""

    if transition == "cut" or len(paths) == 1:
        _ffmpeg_concat_simple(paths, output)
    else:
        try:
            _ffmpeg_concat_xfade(paths, output, transition, t_dur, fps)
        except Exception:
            # xfade can fail on some clip combinations (variable fps, etc.)
            # Fall back to simple concat
            _ffmpeg_concat_simple(paths, output)

    # Mix in background music as a post-processing step
    if music_path and Path(music_path).exists() and output.exists():
        _ffmpeg_add_music(output, music_path, music_vol)

    return output


def _ffmpeg_concat_simple(paths: list[Path], output: Path) -> None:
    """Concatenate using the concat demuxer — fastest, no re-encoding."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p.as_posix()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(list_file).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")


def _ffmpeg_concat_xfade(
    paths: list[Path], output: Path, transition: str, t_dur: float, fps: int
) -> None:
    """Concatenate with xfade transitions. Re-encodes all clips."""
    n = len(paths)
    # Build the complex filter expression
    # Each clip needs its duration to compute xfade offset
    durations = [_get_duration(p) for p in paths]

    # Build input args
    inputs = []
    for p in paths:
        inputs += ["-i", str(p)]

    # Build filter chain
    xfade_type = "dissolve" if transition == "dissolve" else "fade"
    filter_parts: list[str] = []
    video_labels: list[str] = [f"[{i}:v]" for i in range(n)]

    current_label = video_labels[0]
    offset = durations[0] - t_dur

    for i in range(1, n):
        next_label = video_labels[i]
        out_label = f"[xf{i}]" if i < n - 1 else "[vout]"
        filter_parts.append(
            f"{current_label}{next_label}xfade=transition={xfade_type}"
            f":duration={t_dur}:offset={offset:.3f}{out_label}"
        )
        offset += durations[i] - t_dur
        current_label = out_label

    # Audio concat (simple)
    audio_labels = [f"[{i}:a]" for i in range(n)]
    filter_parts.append(f"{''.join(audio_labels)}concat=n={n}:v=0:a=1[aout]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xfade failed:\n{result.stderr}")


def _ffmpeg_add_music(video: Path, music: Path, volume: float) -> None:
    """Mix background music into an existing video file in-place."""
    tmp = video.with_suffix(".music_tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(music),
        "-filter_complex",
        f"[0:a][1:a]amix=inputs=2:duration=first:weights=1 {volume}[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        tmp.replace(video)
    else:
        tmp.unlink(missing_ok=True)


def _get_duration(path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 5.0   # fallback assumption
