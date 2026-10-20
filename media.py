"""
Media analysis tools for the AI agent.

Provides capabilities to analyze images, audio, and video files by:
- Images: Encodes to base64 for vision-capable models, or extracts metadata/OCR text
- Audio: Transcribes speech using OpenAI Whisper (local) or extracts metadata
- Video: Extracts key frames + audio transcript for comprehensive analysis
"""
from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional


# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".m4v"}


def _file_exists(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p


def _get_media_type(path: Path) -> str:
    """Detect whether a file is image, audio, or video."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
    return "unknown"


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def encode_image_base64(path: str | Path) -> tuple[str, str]:
    """
    Read an image file and return (base64_data, mime_type).
    Used to send images to vision-capable models via Ollama.
    """
    p = _file_exists(path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        ext = p.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".bmp": "image/bmp", ".webp": "image/webp",
            ".tiff": "image/tiff", ".tif": "image/tiff",
            ".svg": "image/svg+xml",
        }
        mime = mime_map.get(ext, "image/png")
    data = p.read_bytes()
    return base64.b64encode(data).decode("utf-8"), mime


def get_image_info(path: str | Path) -> str:
    """Get basic image metadata without requiring PIL."""
    p = _file_exists(path)
    size_kb = p.stat().st_size / 1024
    info = f"Image: {p.name}\nSize: {size_kb:.1f} KB\nFormat: {p.suffix.upper()}"

    try:
        from PIL import Image
        with Image.open(p) as img:
            info += f"\nDimensions: {img.width}x{img.height}"
            info += f"\nMode: {img.mode}"
            if hasattr(img, "info"):
                exif = img.info.get("exif")
                if exif:
                    info += "\nEXIF data: present"
    except ImportError:
        info += "\n(Install Pillow for detailed image metadata: pip install Pillow)"

    return info


def analyze_image(path: str | Path, prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
    """
    Prepare an image for vision model analysis.
    Returns a dict with base64 data and metadata that the agent can
    inject into a vision-capable message.
    """
    p = _file_exists(path)
    b64_data, mime = encode_image_base64(p)
    metadata = get_image_info(p)

    return {
        "type": "image",
        "path": str(p),
        "base64": b64_data,
        "mime_type": mime,
        "metadata": metadata,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def _ffprobe_metadata(path: Path) -> str:
    """Extract audio/video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path)
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            streams = data.get("streams", [])
            lines = [f"Format: {fmt.get('format_long_name', 'unknown')}"]
            duration = fmt.get("duration")
            if duration:
                secs = float(duration)
                mins, secs = divmod(int(secs), 60)
                hours, mins = divmod(mins, 60)
                lines.append(f"Duration: {hours:02d}:{mins:02d}:{secs:02d}")
            bitrate = fmt.get("bit_rate")
            if bitrate:
                lines.append(f"Bitrate: {int(bitrate) // 1000} kbps")
            for s in streams:
                codec = s.get("codec_type", "unknown")
                codec_name = s.get("codec_long_name", s.get("codec_name", "?"))
                if codec == "audio":
                    sr = s.get("sample_rate", "?")
                    ch = s.get("channels", "?")
                    lines.append(f"Audio: {codec_name}, {sr} Hz, {ch} channels")
                elif codec == "video":
                    w = s.get("width", "?")
                    h = s.get("height", "?")
                    fps = s.get("r_frame_rate", "?")
                    lines.append(f"Video: {codec_name}, {w}x{h}, {fps} fps")
            tags = fmt.get("tags", {})
            if tags:
                for k, v in list(tags.items())[:10]:
                    lines.append(f"Tag {k}: {v}")
            return "\n".join(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return f"File: {path.name}, Size: {path.stat().st_size / 1024:.1f} KB"


def transcribe_audio(path: str | Path, model_size: str = "base") -> str:
    """
    Transcribe audio using OpenAI Whisper (local).
    Falls back to metadata-only if whisper is not installed.
    """
    p = _file_exists(path)
    metadata = _ffprobe_metadata(p)

    # Try local whisper first
    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(str(p))
        transcript = result.get("text", "").strip()
        language = result.get("language", "unknown")

        segments_info = ""
        segments = result.get("segments", [])
        if segments and len(segments) > 1:
            segments_info = f"\nSegments: {len(segments)}"
            for seg in segments[:20]:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                segments_info += f"\n  [{start:.1f}s - {end:.1f}s] {text}"
            if len(segments) > 20:
                segments_info += f"\n  ... and {len(segments) - 20} more segments"

        return (
            f"=== Audio Analysis ===\n"
            f"{metadata}\n"
            f"Language: {language}\n\n"
            f"=== Transcript ===\n"
            f"{transcript}\n"
            f"{segments_info}"
        )
    except ImportError:
        pass

    # Try faster-whisper
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(p))
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(
                f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}"
            )
        transcript = "\n".join(transcript_parts) if transcript_parts else "(no speech detected)"

        return (
            f"=== Audio Analysis ===\n"
            f"{metadata}\n"
            f"Language: {info.language} (probability: {info.language_probability:.1%})\n\n"
            f"=== Transcript ===\n"
            f"{transcript}"
        )
    except ImportError:
        pass

    # Try speech_recognition as last resort
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        # Convert to WAV if needed
        audio_path = str(p)
        if p.suffix.lower() != ".wav":
            try:
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_tmp.close()
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", wav_tmp.name],
                    capture_output=True, timeout=60,
                )
                audio_path = wav_tmp.name
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return (
                    f"=== Audio Analysis ===\n{metadata}\n\n"
                    "Cannot transcribe: ffmpeg not found for format conversion.\n"
                    "Install ffmpeg or use a .wav file."
                )

        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            transcript = "(speech not recognized)"
        except sr.RequestError as e:
            transcript = f"(recognition service error: {e})"

        return (
            f"=== Audio Analysis ===\n"
            f"{metadata}\n\n"
            f"=== Transcript (Google Speech Recognition) ===\n"
            f"{transcript}"
        )
    except ImportError:
        pass

    return (
        f"=== Audio Analysis ===\n"
        f"{metadata}\n\n"
        "No speech recognition library found.\n"
        "Install one of:\n"
        "  pip install openai-whisper    (best quality, runs locally)\n"
        "  pip install faster-whisper    (faster, runs locally)\n"
        "  pip install SpeechRecognition (uses Google API)"
    )


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def _extract_video_frames(path: Path, max_frames: int = 8) -> List[str]:
    """
    Extract key frames from a video using ffmpeg.
    Returns a list of base64-encoded JPEG images.
    """
    frames = []
    try:
        # Get video duration
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0

        if duration <= 0:
            return frames

        # Calculate frame extraction interval
        interval = max(duration / (max_frames + 1), 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(max_frames):
                timestamp = interval * (i + 1)
                if timestamp >= duration:
                    break
                out_path = Path(tmpdir) / f"frame_{i:04d}.jpg"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-ss", str(timestamp),
                        "-i", str(path), "-frames:v", "1",
                        "-q:v", "3", str(out_path),
                    ],
                    capture_output=True, timeout=15,
                )
                if out_path.exists() and out_path.stat().st_size > 0:
                    b64 = base64.b64encode(out_path.read_bytes()).decode("utf-8")
                    frames.append(b64)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return frames


def _extract_audio_from_video(path: Path) -> Optional[str]:
    """Extract audio track from video to a temp WAV file."""
    try:
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ar", "16000", "-ac", "1", wav_tmp.name],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and Path(wav_tmp.name).stat().st_size > 0:
            return wav_tmp.name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def analyze_video(path: str | Path, max_frames: int = 8) -> Dict[str, Any]:
    """
    Analyze a video file by extracting key frames and transcribing audio.
    Returns a dict with frames (base64), transcript, and metadata.
    """
    p = _file_exists(path)
    metadata = _ffprobe_metadata(p)

    # Extract key frames
    frames = _extract_video_frames(p, max_frames=max_frames)

    # Extract and transcribe audio
    transcript = ""
    audio_path = _extract_audio_from_video(p)
    if audio_path:
        try:
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            transcript = f"(Audio transcription failed: {e})"
        finally:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:
                pass

    return {
        "type": "video",
        "path": str(p),
        "metadata": metadata,
        "frames": frames,
        "frame_count": len(frames),
        "transcript": transcript,
    }


# ---------------------------------------------------------------------------
# Unified media analysis entry point
# ---------------------------------------------------------------------------

def analyze_media(path: str | Path, prompt: str = "") -> str:
    """
    Unified entry point: auto-detect media type and return a text analysis.
    For images, returns metadata + base64 marker.
    For audio, returns transcript.
    For video, returns metadata + frame descriptions + transcript.
    """
    p = _file_exists(path)
    media_type = _get_media_type(p)

    if media_type == "image":
        info = get_image_info(p)
        return (
            f"=== Image Analysis ===\n"
            f"{info}\n\n"
            f"[IMAGE_LOADED: {p.name} — ready for vision analysis]\n"
            f"Use the describe_image tool to get AI-powered description."
        )

    elif media_type == "audio":
        return transcribe_audio(p)

    elif media_type == "video":
        result = analyze_video(p)
        parts = [
            f"=== Video Analysis ===",
            result["metadata"],
            f"\nExtracted {result['frame_count']} key frames for analysis.",
        ]
        if result["transcript"]:
            parts.append(f"\n{result['transcript']}")
        else:
            parts.append("\n(No audio transcript available)")
        return "\n".join(parts)

    else:
        return f"Unknown media type for file: {p.name} (extension: {p.suffix})"
