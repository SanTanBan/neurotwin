"""Encode the NeuroTwin demo video from a prepared kit (1080p frames, narration, timings) on a CPU session.

The kit is made locally by `python tools/make_video.py prepare video/scenes.json video/build/kit` (the narration
uses the Windows speech engine) and uploaded as the private dataset nt-video-kit. The encoding is the same as
`python tools/make_video.py encode`.
"""
import glob
import json
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "moviepy==2.1.2"], check=True)

from moviepy import AudioFileClip, ImageClip, concatenate_videoclips  # noqa: E402

WORK = "/kaggle/working" if (os.name != "nt" and os.path.isdir("/kaggle/working")) else "."
KIT = os.path.dirname(glob.glob(os.path.join("/kaggle/input" if WORK != "." else ".", "**", "scenes.json"),
                                recursive=True)[0])


def main():
    spec = json.load(open(os.path.join(KIT, "scenes.json")))
    clips = []
    for sc in spec["scenes"]:
        dur = sc["duration"]
        audio = None
        if "audio" in sc:                     # stop just short of the audio's end (it ends in silence):
            audio = AudioFileClip(os.path.join(KIT, sc["audio"]))      # reading exactly to the end can overrun
            dur = min(dur, audio.duration - 0.04)    # prepare already shortened it
        clip = ImageClip(os.path.join(KIT, sc["image"]), duration=dur)
        if audio is not None:
            clip = clip.with_audio(audio.subclipped(0, dur))
        clips.append(clip.with_duration(dur))
    out = os.path.join(WORK, "neurotwin_demo.mp4")
    concatenate_videoclips(clips, method="compose").write_videofile(out, fps=24, codec="libx264",
                                                                    audio_codec="aac", logger=None)
    print(f"wrote {out}: {sum(c.duration for c in clips):.1f} s, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
