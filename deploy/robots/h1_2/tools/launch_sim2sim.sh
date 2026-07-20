#!/bin/bash
# ============================================================
# launch_sim2sim.sh — one-command H1-2 MuJoCo sim2sim bring-up.
#
#   launch_sim2sim.sh             MuJoCo sim + controller        (2 Konsole tabs)
#   launch_sim2sim.sh --sidecar   + balance_metrics sidecar      (+1 tab, tv env)
#   launch_sim2sim.sh --tau       + tau_monitor (per-motor torque, +1 tab, tv env)
#   launch_sim2sim.sh --all       sim + controller + sidecar + tau
#
# Builds the sim and the controller first (incremental — no-op if up to date).
# All run on `lo`, DDS domain 0 (matches simulate/config.yaml).
#
#   -h | --help   show this header
# ============================================================
set -u

SIDECAR=0
TAU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sidecar) SIDECAR=1; shift ;;
    --tau)     TAU=1; shift ;;
    --all)     SIDECAR=1; TAU=1; shift ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1 (use --help)"; exit 1 ;;
  esac
done

REPOS="${REPOS:-$HOME/Projects/robot_projects/repos}"
MJ_BUILD="$REPOS/unitree_mujoco/simulate/build"
CTRL_DIR="$REPOS/unitree_rl_lab/deploy/robots/h1_2"
CTRL_BUILD="$CTRL_DIR/build"
SIDECAR_DIR="$CTRL_DIR/tools"
TAU_MON="$REPOS/unitree_mujoco/mujoco_sim/tools/tau_monitor.py"
TV_ENV="${TV_ENV:-tv}"

CONDA_BASE="$(conda info --base 2>/dev/null)"
[[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- build (incremental) -----------------------------------------------------
build_target() {  # $1 = build dir, $2 = label
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

# ---- inner tab scripts -------------------------------------------------------
STAMP="$(date +%H%M%S)"
cat > /tmp/sim2sim_mujoco.sh << EOF
#!/bin/bash
echo ">>> [TAB 1] MuJoCo sim — h1_2 on lo, domain 0"
echo "    keys: J/K/L arm presets | [ ] \\ payload | Z zero metrics | 9/arrows band"
cd "$MJ_BUILD"
./unitree_mujoco 2>&1 | tee /tmp/sim2sim_mujoco_$STAMP.log
exec bash
EOF

cat > /tmp/sim2sim_ctrl.sh << EOF
#!/bin/bash
echo ">>> [TAB 2] h1_2 controller — waiting 2s for the sim ..."
sleep 2
cd "$CTRL_BUILD"
./h1_2_ctrl --network lo 2>&1 | tee /tmp/sim2sim_ctrl_$STAMP.log
exec bash
EOF

cat > /tmp/sim2sim_sidecar.sh << EOF
#!/bin/bash
echo ">>> [TAB 3] balance_metrics sidecar (tv env) — waiting 3s ..."
sleep 3
[[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $TV_ENV
cd "$SIDECAR_DIR"
python3 -u balance_metrics.py --iface lo --domain 0 2>&1 | tee /tmp/sim2sim_sidecar_$STAMP.log
exec bash
EOF
cat > /tmp/sim2sim_tau.sh << EOF
#!/bin/bash
echo ">>> [TAB] tau_monitor — per-motor torque (hip-roll m2/m8 squeeze), waiting 3s ..."
[[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $TV_ENV
sleep 3
python3 -u "$TAU_MON" --iface lo --domain 0 2>&1 | tee /tmp/sim2sim_tau_$STAMP.log
exec bash
EOF
chmod +x /tmp/sim2sim_mujoco.sh /tmp/sim2sim_ctrl.sh /tmp/sim2sim_sidecar.sh /tmp/sim2sim_tau.sh

# ---- launch Konsole tabs -----------------------------------------------------
{
  echo "title: mujoco;; command: /tmp/sim2sim_mujoco.sh"
  echo "title: controller;; command: /tmp/sim2sim_ctrl.sh"
  [[ $SIDECAR -eq 1 ]] && echo "title: sidecar;; command: /tmp/sim2sim_sidecar.sh"
  [[ $TAU -eq 1 ]] && echo "title: tau;; command: /tmp/sim2sim_tau.sh"
} > /tmp/sim2sim_tabs.cfg

NTABS=$((2 + SIDECAR + TAU))
echo ">>> launching Konsole ($NTABS tabs: sim + controller$([[ $SIDECAR -eq 1 ]] && echo ' + sidecar')$([[ $TAU -eq 1 ]] && echo ' + tau')) ..."
konsole --tabs-from-file /tmp/sim2sim_tabs.cfg >/dev/null 2>&1 &
KPID=$!
sleep 2
if ! kill -0 "$KPID" 2>/dev/null; then
  echo "   (tabs-from-file unavailable; spawning separate windows)"
  konsole -e bash /tmp/sim2sim_mujoco.sh >/dev/null 2>&1 &
  sleep 1
  konsole -e bash /tmp/sim2sim_ctrl.sh >/dev/null 2>&1 &
  [[ $SIDECAR -eq 1 ]] && { sleep 1; konsole -e bash /tmp/sim2sim_sidecar.sh >/dev/null 2>&1 & }
  [[ $TAU -eq 1 ]] && { sleep 1; konsole -e bash /tmp/sim2sim_tau.sh >/dev/null 2>&1 & }
fi

cat << EOF

================================================================
  sim2sim launched.
    TAB 1 mujoco     — disable the elastic band for free balance
    TAB 2 controller — LT+Up (FixStand) -> RB+Y/A/X.. (Balance policy)
$([[ $SIDECAR -eq 1 ]] && echo "    TAB sidecar      — metrics print + publish to the sim HUD (bottom-left)")
$([[ $TAU -eq 1 ]] && echo "    TAB tau          — per-motor torque; watch hipR m2/m8 on the GRIP scene (anti-squeeze verdict)")
  HUD: top-left = policy/gains, bottom-left = payload$([[ $SIDECAR -eq 1 ]] && echo ' + metrics')
  Logs: /tmp/sim2sim_*_$STAMP.log
================================================================
EOF
