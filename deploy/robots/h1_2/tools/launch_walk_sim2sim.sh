#!/bin/bash
# ============================================================
# launch_walk_sim2sim.sh — one-command H1-2 WALK sim2sim bring-up.
#
#   3 Konsole tabs: MuJoCo sim (desk-free scene) + controller + keyboard teleop
#
# Scene: auto-sets robot_scene to scene_comx06.xml (NO desk — walking needs
# open floor). The balance launcher's desk default is restored by just running
# launch_sim2sim.sh again (it does not manage the scene) or run_mujoco_desk_line.sh.
#
# Fly it:
#   TAB 2 (controller stdin: 'fsm FixStand' then 'fsm Walk' — names, digits are positional)
#     LT+Up   -> FixStand           (via sim window key or rt/fsm_cmd digits)
#     RB+left -> Walk               (id 10)
#   TAB 3 (teleop) — focus it and drive:
#     arrows = vx/vy   Q/E = yaw   Space = zero   Tab = sticky   Esc = quit
#
#   -h | --help   show this header
# ============================================================
set -u

case "${1:-}" in -h|--help) sed -n '2,19p' "$0" | sed 's/^# \?//'; exit 0 ;; esac

REPOS="${REPOS:-$HOME/Projects/robot_projects/repos}"
MJ_DIR="$REPOS/unitree_mujoco"
MJ_BUILD="$MJ_DIR/simulate/build"
CTRL_DIR="$REPOS/unitree_rl_lab/deploy/robots/h1_2"
CTRL_BUILD="$CTRL_DIR/build"
TOOLS_DIR="$CTRL_DIR/tools"
TV_ENV="${TV_ENV:-tv}"

CONDA_BASE="$(conda info --base 2>/dev/null)"
[[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- scene: desk-free comx06 (walking wants open floor) ------------------------
MJ_CFG="$MJ_DIR/simulate/config.yaml"
if ! grep -qE 'robot_scene: "scene_comx06.xml"' "$MJ_CFG"; then
  sed -i 's/robot_scene: "[^"]*"/robot_scene: "scene_comx06.xml"/' "$MJ_CFG"
  echo ">>> [scene] robot_scene -> scene_comx06.xml (comx06 body, rigid floor, NO desk)"
fi

# ---- build (incremental) -------------------------------------------------------
build_target() {
  local bdir="$1" label="$2"
  echo ">>> building $label ..."
  mkdir -p "$bdir"
  if [[ ! -f "$bdir/CMakeCache.txt" ]]; then
    ( cd "$bdir" && cmake .. ) || { echo "!! cmake failed: $label"; exit 1; }
  fi
  ( cd "$bdir" && make -j4 ) || { echo "!! build failed: $label"; exit 1; }
}
build_target "$MJ_BUILD"   "unitree_mujoco (sim)"
build_target "$CTRL_BUILD" "h1_2_ctrl (controller)"

# ---- inner tab scripts ----------------------------------------------------------
STAMP="$(date +%H%M%S)"
cat > /tmp/walk_mujoco.sh << EOF
#!/bin/bash
echo ">>> [TAB 1] MuJoCo sim — h1_2 walk scene (no desk), lo/domain 0"
echo "    band keys 9/arrows live HERE; release the band before walking off"
cd "$MJ_BUILD"
./unitree_mujoco 2>&1 | tee /tmp/walk_mujoco_$STAMP.log
exec bash
EOF

cat > /tmp/walk_ctrl.sh << EOF
#!/bin/bash
echo ">>> [TAB 2] h1_2 controller — waiting 2s for the sim ..."
echo "    stdin: 'fsm FixStand' then 'fsm Walk' ('fsm Passive' to drop). Digit keys are POSITIONAL — use names."
sleep 2
cd "$CTRL_BUILD"
./h1_2_ctrl --network lo 2>&1 | tee /tmp/walk_ctrl_$STAMP.log
exec bash
EOF

cat > /tmp/walk_teleop.sh << EOF
#!/bin/bash
echo ">>> [TAB 3] walk teleop (tv env) — waiting 3s ..."
[[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $TV_ENV
sleep 3
cd "$TOOLS_DIR"
python3 -u walk_teleop.py
exec bash
EOF
chmod +x /tmp/walk_mujoco.sh /tmp/walk_ctrl.sh /tmp/walk_teleop.sh

# ---- launch Konsole tabs ---------------------------------------------------------
{
  echo "title: mujoco;; command: /tmp/walk_mujoco.sh"
  echo "title: controller;; command: /tmp/walk_ctrl.sh"
  echo "title: teleop;; command: /tmp/walk_teleop.sh"
} > /tmp/walk_tabs.cfg

echo ">>> launching Konsole (3 tabs: sim + controller + teleop) ..."
konsole --tabs-from-file /tmp/walk_tabs.cfg >/dev/null 2>&1 &
KPID=$!
sleep 2
if ! kill -0 "$KPID" 2>/dev/null; then
  echo "   (tabs-from-file unavailable; spawning separate windows)"
  konsole -e bash /tmp/walk_mujoco.sh >/dev/null 2>&1 &
  sleep 1
  konsole -e bash /tmp/walk_ctrl.sh >/dev/null 2>&1 &
  sleep 1
  konsole -e bash /tmp/walk_teleop.sh >/dev/null 2>&1 &
fi

cat << EOF

================================================================
  WALK sim2sim launched.
    TAB 1 mujoco     — scene_comx06.xml, no desk. Band: 9/arrows.
    TAB 2 controller — 'fsm FixStand' then 'fsm Walk'
    TAB 3 teleop     — arrows vx/vy, Q/E yaw, Space zero, Tab sticky
  Contract clamps: vx [-0.3, 1.0]  vy ±0.3  wz ±0.5
  Logs: /tmp/walk_*_$STAMP.log
================================================================
EOF
