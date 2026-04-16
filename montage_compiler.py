"""FFmpeg-based video montage compiler — OpenMontage-style automatic compilation."""

import asyncio
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional


TRANSITION_FILTERS = {
    "fade": "fade=t=in:st=0:d={dur},fade=t=out:st={fade_out}:d={dur}",
    "dissolve": "xfade=transition=dissolve:duration={dur}:offset={offset}",
    "wipe": "xfade=transition=wipeleft:duration={dur}:offset={offset}",
    "slide": "xfade=transition=slideleft:duration={dur}:offset={offset}",
    "zoom": "xfade=transition=zoomin:duration={dur}:offset={offset}",
    "none": "",
}


def get_ffmpeg() -> str:
    """Find FFmpeg executable."""
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        if shutil.which(cmd):
            return cmd
    raise RuntimeError(
        "FFmpeg not found. Install it from https://ffmpeg.org/download.html "
        "and add to PATH."
    )


def get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ffprobe:
        return 3.0  # fallback

    result = subprocess.run(
        [
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_streams", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                return float(stream.get("duration", 3.0))
    except Exception:
        pass
    return 3.0


def get_video_info(path: str) -> dict:
    """Get video width, height, fps."""
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ffprobe:
        return {"width": 512, "height": 512, "fps": 8}

    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "24/1")
                num, den = fps_str.split("/")
                fps = int(num) / int(den)
                return {
                    "width": stream.get("width", 512),
                    "height": stream.get("height", 512),
                    "fps": fps,
                    "duration": float(stream.get("duration", 3.0)),
                }
    except Exception:
        pass
    return {"width": 512, "height": 512, "fps": 8, "duration": 3.0}


class MontageCompiler:
    """Compile multiple video clips into a polished montage."""

    def __init__(self, config: dict):
        self.config = config
        self.transition = config.get("default_transition", "fade")
        self.transition_dur = config.get("transition_duration", 0.5)
        self.resolution = config.get("default_resolution", "1280x720")
        self.fps = config.get("default_fps", 24)
        self.music = config.get("default_music", "")
        self.music_vol = config.get("music_volume", 0.3)

    async def compile(
        self,
        video_paths: list[str],
        output_path: str,
        title: str = "",
        transition: Optional[str] = None,
        resolution: Optional[str] = None,
        fps: Optional[int] = None,
        music_path: Optional[str] = None,
        progress_callback=None,
    ) -> str:
        """Compile videos into a montage. Returns output path."""
        ffmpeg = get_ffmpeg()
        transition = transition or self.transition
        resolution = resolution or self.resolution
        fps = fps or self.fps
        music = music_path or self.music

        w, h = map(int, resolution.split("x"))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(video_paths) == 1:
            return await self._process_single(ffmpeg, video_paths[0], str(output_path), w, h, fps)

        if transition == "none" or not TRANSITION_FILTERS.get(transition):
            return await self._concat_simple(ffmpeg, video_paths, str(output_path), w, h, fps, music)
        else:
            return await self._concat_xfade(ffmpeg, video_paths, str(output_path), w, h, fps, transition, music)

    async def _process_single(self, ffmpeg, input_path, output_path, w, h, fps) -> str:
        """Process a single video — scale and normalize."""
        cmd = [
            ffmpeg, "-y", "-i", str(input_path),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        await self._run(cmd)
        return str(output_path)

    async def _concat_simple(self, ffmpeg, video_paths, output_path, w, h, fps, music) -> str:
        """Simple concatenation without transitions using concat demuxer."""
        # Write concat file
        concat_file = Path(output_path).parent / "_concat_list.txt"
        with open(concat_file, "w") as f:
            for vp in video_paths:
                f.write(f"file '{Path(vp).resolve()}'\n")

        # Build scale filter
        scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"

        if music and Path(music).exists():
            cmd = [
                ffmpeg, "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-i", music,
                "-vf", scale,
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-filter_complex", f"[1:a]volume={self.music_vol}[music];[music]apad[a]",
                "-map", "0:v", "-map", "[a]",
                "-shortest", "-pix_fmt", "yuv420p",
                str(output_path),
            ]
        else:
            cmd = [
                ffmpeg, "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-vf", scale,
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-an",
                str(output_path),
            ]

        await self._run(cmd)
        concat_file.unlink(missing_ok=True)
        return str(output_path)

    async def _concat_xfade(self, ffmpeg, video_paths, output_path, w, h, fps, transition, music) -> str:
        """Concatenation with xfade transitions between clips."""
        # Get durations
        durations = [get_video_duration(vp) for vp in video_paths]
        td = self.transition_dur

        # Build complex filter graph
        n = len(video_paths)
        scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"

        inputs = []
        for vp in video_paths:
            inputs += ["-i", str(vp)]

        # Scale each input
        filter_parts = []
        for i in range(n):
            filter_parts.append(f"[{i}:v]{scale}[v{i}]")

        # Chain xfade
        offset = durations[0] - td
        if n == 2:
            filter_parts.append(
                f"[v0][v1]xfade=transition={transition}:duration={td}:offset={offset:.3f}[outv]"
            )
        else:
            filter_parts.append(
                f"[v0][v1]xfade=transition={transition}:duration={td}:offset={offset:.3f}[xf0]"
            )
            for i in range(2, n):
                offset += durations[i - 1] - td
                src = f"[xf{i-2}]" if i > 2 else "[xf0]"
                dst = "[outv]" if i == n - 1 else f"[xf{i-1}]"
                filter_parts.append(
                    f"{src}[v{i}]xfade=transition={transition}:duration={td}:offset={offset:.3f}{dst}"
                )

        filter_complex = ";".join(filter_parts)

        cmd = [ffmpeg, "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]

        if music and Path(music).exists():
            # Add music as second pass
            temp_out = str(output_path) + ".tmp.mp4"
            cmd[-1] = temp_out
            await self._run(cmd)
            cmd2 = [
                ffmpeg, "-y", "-i", temp_out, "-i", music,
                "-filter_complex", f"[1:a]volume={self.music_vol}[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(output_path),
            ]
            await self._run(cmd2)
            Path(temp_out).unlink(missing_ok=True)
        else:
            await self._run(cmd)

        return str(output_path)

    async def add_title_card(self, video_path: str, title: str, output_path: str) -> str:
        """Add a title card at the beginning of the video."""
        ffmpeg = get_ffmpeg()
        info = get_video_info(video_path)
        w, h = info["width"], info["height"]

        # Generate title card using ffmpeg lavfi
        title_dur = 2.0
        title_file = Path(output_path).parent / "_title_card.mp4"

        cmd1 = [
            ffmpeg, "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={w}x{h}:d={title_dur}",
            "-vf", f"drawtext=text='{title}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(title_file),
        ]
        await self._run(cmd1)

        # Concat title + video
        concat_file = Path(output_path).parent / "_title_concat.txt"
        with open(concat_file, "w") as f:
            f.write(f"file '{title_file.resolve()}'\n")
            f.write(f"file '{Path(video_path).resolve()}'\n")

        cmd2 = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(output_path),
        ]
        await self._run(cmd2)
        title_file.unlink(missing_ok=True)
        concat_file.unlink(missing_ok=True)
        return str(output_path)

    async def _run(self, cmd: list[str]):
        """Run an FFmpeg command asynchronously."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{stderr.decode()[-2000:]}")
