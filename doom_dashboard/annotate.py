"""
Video annotation and MP4 export.

Takes an EpisodeData and renders an annotated MP4 with:
  - Action labels (button names pressed each frame)
  - Reward bar overlay
  - Episode step counter and timing
  - Map name + policy name watermark

Uses imageio + ffmpeg (no display required).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doom_dashboard.rollout import EpisodeData


# ─── font helpers ────────────────────────────────────────────────

def _get_font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


# ─── single-frame annotation ─────────────────────────────────────

_FONT_SMALL = None
_FONT_LARGE = None

def _fonts():
    global _FONT_SMALL, _FONT_LARGE
    if _FONT_SMALL is None:
        _FONT_SMALL = _get_font(13)
        _FONT_LARGE = _get_font(16)
    return _FONT_SMALL, _FONT_LARGE


def annotate_frame(
    frame: np.ndarray,
    action: np.ndarray,
    button_names: list,
    reward: float,
    step: int,
    total_steps: int,
    scenario_name: str,
    policy_name: str,
    total_reward_so_far: float,
) -> np.ndarray:
    """Return an annotated copy of a frame as uint8 H×W×3.

    All overlays are at the TOP so the native video scrubber at the
    bottom is not obscured.
    """
    img = Image.fromarray(frame if frame.shape[2] == 3 else frame[:, :, :3])
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_s, font_l = _fonts()

    row_h = 20
    top_h  = row_h * 2 + 4          # two-row top banner

    # semi-transparent top banner
    draw.rectangle([0, 0, W, top_h], fill=(15, 15, 20, 210))

    # row 1 — actions (green) + step/reward (yellow, right-aligned)
    pressed = [
        b.replace("Button.", "").replace("_", " ")
        for b, a in zip(button_names, action) if a > 0.5
    ]
    action_str = "  ".join(pressed) if pressed else "NOOP"
    draw.text((4, 3), f"▶ {action_str}", fill=(80, 255, 110), font=font_s)

    rw_str    = f"r={reward:+.1f}  Σ={total_reward_so_far:+.1f}  {step}/{total_steps}"
    try:
        rw_w = font_s.getbbox(rw_str)[2]
    except AttributeError:
        rw_w = len(rw_str) * 7
    draw.text((W - rw_w - 4, 3), rw_str, fill=(255, 210, 60), font=font_s)

    # row 2 — map + policy (grey)
    draw.text((4, row_h + 4), f"{scenario_name}  ·  {policy_name}", fill=(160, 160, 180), font=font_s)

    # thin left-edge time progress bar (below top banner only)
    if total_steps > 0:
        avail  = H - top_h
        filled = int((step / total_steps) * avail)
        draw.rectangle([0, top_h, 3, top_h + filled], fill=(80, 200, 120))

    return np.array(img)


# ─── full episode → MP4 ──────────────────────────────────────────

def annotate_and_encode(
    episode: EpisodeData,
    out_path: str,
    fps: int = 30,
    annotate: bool = True,
) -> str:
    """Encode EpisodeData to an annotated MP4.

    Parameters
    ----------
    episode:    EpisodeData from rollout_episode()
    out_path:   Output .mp4 path (parent dir will be created)
    fps:        Output video FPS
    annotate:   If False, write raw frames (no overlays)

    Returns
    -------
    out_path (same as input)
    """
    import imageio

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    total = len(episode.frames)
    cumulative_reward = 0.0

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                 output_params=["-crf", "22", "-preset", "fast"])
    for i, (frame, action, reward) in enumerate(
        zip(episode.frames, episode.actions, episode.rewards)
    ):
        cumulative_reward += reward
        if annotate:
            out_frame = annotate_frame(
                frame=frame,
                action=action,
                button_names=episode.button_names,
                reward=reward,
                step=i + 1,
                total_steps=total,
                scenario_name=episode.scenario_name,
                policy_name=episode.policy_name,
                total_reward_so_far=cumulative_reward,
            )
        else:
            out_frame = frame if frame.shape[2] == 3 else frame[:, :, :3]
        writer.append_data(out_frame)
    writer.close()

    return out_path
