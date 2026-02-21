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
    """Return an annotated copy of a frame as uint8 H×W×3."""
    img = Image.fromarray(frame if frame.shape[2] == 3 else frame[:, :, :3])
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_s, font_l = _fonts()

    # ── top bar: scenario / policy ──
    bar_h = 22
    draw.rectangle([0, 0, W, bar_h], fill=(20, 20, 20, 220))
    draw.text((4, 4), f"Map: {scenario_name}  Policy: {policy_name}", fill=(220, 220, 220), font=font_s)

    # ── bottom bar: actions + reward ──
    bot_y = H - bar_h - 2
    draw.rectangle([0, bot_y, W, H], fill=(20, 20, 20, 220))

    pressed = [button_names[i] for i, a in enumerate(action) if a > 0.5]
    action_str = " | ".join(pressed) if pressed else "NOOP"
    draw.text((4, bot_y + 4), f"A: {action_str}", fill=(100, 255, 100), font=font_s)

    # right side: step counter + cumulative reward
    rw_str = f"r={reward:+.1f}  Σ={total_reward_so_far:+.1f}  t={step}/{total_steps}"
    # right-align
    try:
        rw_bbox = font_s.getbbox(rw_str)
        rw_w = rw_bbox[2] - rw_bbox[0]
    except AttributeError:
        rw_w = len(rw_str) * 7
    draw.text((W - rw_w - 4, bot_y + 4), rw_str, fill=(255, 200, 80), font=font_s)

    # ── thin reward bar on left edge ──
    if total_steps > 0:
        prog = int((step / total_steps) * (H - 2 * bar_h))
        draw.rectangle([0, bar_h, 4, bar_h + prog], fill=(80, 200, 120))

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
