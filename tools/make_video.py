"""Assemble the demo video from a scene list.

Each scene is an image (or a video clip) plus narration text. The narration is synthesised locally, and each
scene lasts as long as its narration (plus a small pad), so the video stays in sync without manual timing.

    python tools/make_video.py video/scenes.json video/build/demo.mp4

scenes.json: {"voice": "...", "rate": 0, "scenes": [{"image": "runs/figs/x.png", "text": "..."}, ...]}
A scene may use "video" instead of "image" (e.g. a screen recording), and may set "duration" to override.
"""
import json
import os
import subprocess
import sys
import wave

from moviepy import (AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoFileClip,
                     concatenate_videoclips)
from PIL import Image

SIZE = (1920, 1080)
PAD_S = 0.6
BG = (255, 255, 255)


def letterbox_image(path, out_path):
    """Centre an image on the 1080p canvas with PIL: cheaper and far lighter than clip compositing."""
    im = Image.open(path).convert("RGB")
    scale = min(SIZE[0] / im.width, SIZE[1] / im.height)
    small = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", SIZE, BG)
    canvas.paste(small, ((SIZE[0] - small.width) // 2, (SIZE[1] - small.height) // 2))
    canvas.save(out_path)
    return out_path


def fit_canvas(clip, duration):
    """Letterbox a video clip onto the canvas (images take the cheaper PIL path)."""
    scale = min(SIZE[0] / clip.w, SIZE[1] / clip.h)
    inner = clip.resized(scale).with_position("center").with_duration(duration)
    bg = ColorClip(SIZE, color=BG, duration=duration)
    return CompositeVideoClip([bg, inner], size=SIZE).with_duration(duration)


def synthesise(text, path, voice, rate, pad=PAD_S):
    """Synthesise narration and append `pad` seconds of silence, so audio and scene lengths match exactly."""
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "tools/tts.ps1",
                    "-Text", text, "-Out", path, "-Voice", voice, "-Rate", str(rate)], check=True)
    with wave.open(path, "rb") as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    n_pad = int(pad * params.framerate)
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(frames + b"\x00" * (n_pad * params.sampwidth * params.nchannels))
    return (params.nframes + n_pad) / params.framerate


def build(spec_path, out_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    voice, rate = spec.get("voice", "Microsoft Zira Desktop"), spec.get("rate", 0)
    tmp = os.path.join(os.path.dirname(out_path) or ".", "audio")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    clips, total = [], 0.0
    for i, sc in enumerate(spec["scenes"]):
        wav = os.path.join(tmp, f"scene{i:02d}.wav")
        dur = synthesise(sc["text"], wav, voice, rate) if sc.get("text") else sc.get("duration", 3.0)
        dur = sc.get("duration", dur)
        if "video" in sc:
            src = VideoFileClip(sc["video"])
            clip = fit_canvas(src.subclipped(0, min(dur, src.duration)), dur)
        else:
            framed = letterbox_image(sc["image"], os.path.join(tmp, f"scene{i:02d}.png"))
            clip = ImageClip(framed, duration=dur)         # already on the 1080p canvas
        if sc.get("text"):
            clip = clip.with_audio(AudioFileClip(wav))
        clips.append(clip.with_duration(dur))
        total += dur
        print(f"scene {i}: {dur:.1f}s  {sc.get('image') or sc.get('video')}")
    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    print(f"wrote {out_path}  ({total:.1f}s total; the competition limit is 300 s)")
    if total > 300:
        print("WARNING: over the 5-minute limit")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "video/build/demo.mp4")
