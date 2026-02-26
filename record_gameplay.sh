#!/bin/bash
# record_gameplay.sh — Record your own Doom deathmatch gameplay
# Usage: bash record_gameplay.sh [scenario] [minutes] [bots] [name]
#   scenario: deathmatch_compact (default), deathmatch_nomonsters
#   minutes:  timelimit in minutes (default: 5)
#   bots:     number of bots (default: 2)
#   name:     optional session name
#
# IMPORTANT: Run this from a desktop session (physical, VNC, or remote desktop).
# You need a visible window to play! SSH alone won't work.
#
# Saves to human_demos/:
#   <name>.npz    — frames, actions, rewards, game_vars (for training)
#   <name>.mp4    — annotated replay video
#   <name>.meta.json — metadata
set -e
cd /share/NFS/u/wendler/code/doom-dashboard

# OpenAL for sound (built from source in ~/.local)
export LD_LIBRARY_PATH="/share/NFS/u/wendler/.local/lib:${LD_LIBRARY_PATH:-}"

# X display — must be set to a real display
if [ -z "$DISPLAY" ]; then
    echo "ERROR: No DISPLAY set. Run this from a desktop session (VNC, physical, etc.)"
    echo "  If you have a display, set it: export DISPLAY=:0"
    exit 1
fi

SCENARIO="${1:-deathmatch_compact}"
TIMELIMIT="${2:-5}"
BOTS="${3:-2}"
NAME="${4:-}"

echo "=== Doom Gameplay Recorder ==="
echo "  Scenario: $SCENARIO"
echo "  Timelimit: ${TIMELIMIT} minutes"
echo "  Bots: $BOTS"
echo "  Display: $DISPLAY"
echo "  Output: human_demos/"
echo ""
echo "Controls:"
echo "  WASD or arrows  = move"
echo "  Mouse            = aim & look"
echo "  Left click       = shoot"
echo "  1-6              = weapon select"
echo ""
echo "  Click into the game window to capture keyboard/mouse!"
echo "  ESC -> Options -> Customize Controls to rebind keys"
echo ""
echo "Starting in 3 seconds..."
sleep 3

NAME_ARG=""
[ -n "$NAME" ] && NAME_ARG="--name $NAME"

uv run python -m doom_dashboard.cli record-human-demo \
    --scenario "$SCENARIO" \
    --map map01 \
    --timelimit "$TIMELIMIT" \
    --bots "$BOTS" \
    --resolution RES_1280X720 \
    --render-hud \
    --frame-skip 1 \
    $NAME_ARG

echo ""
echo "Recording saved to human_demos/"
echo "Files:"
ls -lh human_demos/*"${NAME:-human_}"* 2>/dev/null | tail -10
