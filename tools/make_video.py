"""Assemble the demo video from a scene list.

Each scene is an image (or a video clip) plus narration text. The narration is synthesised locally with the
Windows speech engine, and each scene lasts as long as its narration (plus a small pad), so the video stays in
sync without manual timing. Preparing (speech, letterboxing) and encoding are separate steps, so the encoding can
run elsewhere — for example on a Kaggle CPU session when the laptop is busy.

    python tools/make_video.py prepare video/scenes.json video/build/kit     # speech + 1080p frames + timings
    python tools/make_video.py encode video/build/kit video/build/neurotwin_demo.mp4
    python tools/make_video.py all video/scenes.json video/build/neurotwin_demo.mp4

scenes.json: {"voice": "...", "rate": 0, "scenes": [{"image": "runs/figs/x.png", "text": "..."}, ...]}
A scene may use "video" instead of "image" (e.g. a screen recording), and may set "duration" to override.
"""
import json
import os
import shutil
import subprocess
import sys
import wave

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


def prepare(spec_path, kit):
    """Write a self-contained kit: letterboxed frames, narration WAVs and kit/scenes.json with the durations."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    voice, rate = spec.get("voice", "Microsoft Zira Desktop"), spec.get("rate", 0)
    os.makedirs(kit, exist_ok=True)
    scenes, total = [], 0.0
    for i, sc in enumerate(spec["scenes"]):
        out = {}
        if "video" in sc:
            out["video"] = f"scene{i:02d}{os.path.splitext(sc['video'])[1]}"
            shutil.copy(sc["video"], os.path.join(kit, out["video"]))
        else:
            out["image"] = f"scene{i:02d}.png"
            letterbox_image(sc["image"], os.path.join(kit, out["image"]))
        dur = sc.get("duration", 3.0)
        if sc.get("text"):
            out["audio"] = f"scene{i:02d}.wav"
            dur = sc.get("duration", synthesise(sc["text"], os.path.join(kit, out["audio"]), voice, rate))
        out["duration"] = dur
        scenes.append(out)
        total += dur
        print(f"scene {i}: {dur:.1f}s  {sc.get('image') or sc.get('video')}")
    json.dump({"scenes": scenes}, open(os.path.join(kit, "scenes.json"), "w"), indent=1)
    print(f"kit in {kit}: {len(scenes)} scenes, {total:.1f} s (the competition limit is 300 s)")
    if total > 300:
        print("WARNING: over the 5-minute limit")


def encode(kit, out_path):
    """Encode a prepared kit into a 1080p, 24-fps H.264/AAC MP4."""
    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips
    clips = []
    for sc in json.load(open(os.path.join(kit, "scenes.json")))["scenes"]:
        dur = sc["duration"]
        if "video" in sc:
            src = VideoFileClip(os.path.join(kit, sc["video"]))
            scale = min(SIZE[0] / src.w, SIZE[1] / src.h)
            inner = src.subclipped(0, min(dur, src.duration)).resized(scale).with_position("center")
            clip = CompositeVideoClip([ColorClip(SIZE, color=BG, duration=dur), inner], size=SIZE)
        else:
            clip = ImageClip(os.path.join(kit, sc["image"]), duration=dur)    # already on the 1080p canvas
        if "audio" in sc:                     # stop just short of the audio's end (it ends in silence):
            audio = AudioFileClip(os.path.join(kit, sc["audio"]))      # reading exactly to the end can overrun
            dur = min(dur, audio.duration) - 0.04
            clip = clip.with_audio(audio.subclipped(0, dur))
        clips.append(clip.with_duration(dur))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    concatenate_videoclips(clips, method="compose").write_videofile(out_path, fps=24, codec="libx264",
                                                                    audio_codec="aac", logger=None)
    print(f"wrote {out_path} ({sum(c.duration for c in clips):.1f} s)")


if __name__ == "__main__":
    cmd, src, dst = sys.argv[1:4]
    if cmd == "prepare":
        prepare(src, dst)
    elif cmd == "encode":
        encode(src, dst)
    else:
        kit = os.path.join(os.path.dirname(dst) or ".", "kit")
        prepare(src, kit)
        encode(kit, dst)
