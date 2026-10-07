"""HipsArm-DR -- the HipsArm trunk (p14g_hips_armfull's recipe) with the domain-randomization hooks (2026-10-07, p16dr wave).

The trunk itself changes NOTHING a policy can feel: the action term becomes mdp.RobustJointPositionAction (same scale / default
offset / clip; encoder subtraction only when an encoder event exists; lag 0) and the policy's joint_pos_rel becomes
mdp.joint_pos_rel_encoder (identical values while the offset buffer is zero). The critic keeps the clean joint_pos_rel. The DR arrives
as genes, one lever each; rows compose them (see mdp/dr.py for the encoder-offset mechanism).

Every trunk has run with DR effectively OFF since p6 (friction 1.0 fixed, base mass +0, gain scale 1.0; only the hand payload and
the pushes), so each gene below is a range on an event that already exists at its nominal value, except the encoder offset (new) and
the torso CoM (p12g_comdr precedent, x only, never on hardware).
  gene_enc03 / gene_enc05  encoder offset +-0.03 / +-0.05 rad per LEG + torso joint, drawn per reset (the MIT baselines: 0.015-0.05)
  gene_com                 torso_link CoM +-2.5 cm in x AND y at startup (torso = 34 % of the mass -> +-8.6 mm whole-body)
  gene_mass                torso mass -0.5..+0.5 kg at startup (the mass is pinned to ~77 kg +-0.2; the band is the residual)
  gene_fric                static / dynamic friction 0.6..1.2 (sole on cement unknown and it should not matter)
  gene_gains               PD stiffness and damping scaled 0.85..1.15 per reset
  gene_lag                 action lag 0..1 policy step (0-20 ms) per reset -- the comms latency is ~1-4 ms, so this is a robustness
                           probe, not a model of the robot
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_hipsarm import _make_hipsarm
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command

DR_LEG_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint"]


def _make_hipsarm_dr(cfg):
    old = cfg.actions.JointPositionAction
    cfg.actions.JointPositionAction = mdp.RobustJointPositionActionCfg(
        asset_name=old.asset_name, joint_names=old.joint_names, scale=old.scale, offset=old.offset,
        use_default_offset=old.use_default_offset, clip=old.clip, preserve_order=old.preserve_order, max_lag_steps=0,
    )
    cfg.observations.policy.joint_pos_rel.func = mdp.joint_pos_rel_encoder
    A = cfg.actions.JointPositionAction
    assert A.class_type is mdp.RobustJointPositionAction and A.scale == 0.25 and A.use_default_offset is True and A.clip is None and A.max_lag_steps == 0, "[HipsArm-DR] action term = Robust, same affine, lag 0"
    assert cfg.observations.policy.joint_pos_rel.func is mdp.joint_pos_rel_encoder and cfg.observations.critic.joint_pos_rel.func is mdp.joint_pos_rel, "[HipsArm-DR] policy sees the encoder frame, critic the true one"
    ev = cfg.events
    assert tuple(ev.physics_material.params["static_friction_range"]) == (1.0, 1.0) and tuple(ev.add_base_mass.params["mass_distribution_params"]) == (0.0, 0.0) and tuple(ev.randomize_motor_strength.params["stiffness_distribution_params"]) == (1.0, 1.0), "[HipsArm-DR] trunk DR nominal (genes open the ranges)"
    assert getattr(ev, "encoder_offset", None) is None and getattr(ev, "randomize_torso_com", None) is None, "[HipsArm-DR] no encoder / CoM event in the trunk"
    assert cfg.rewards.upright_bonus.params["std"] == 0.05 and cfg.rewards.joint_deviation_hips.weight == -1.0, "[HipsArm-DR] HipsArm recipe underneath"


# ── genes ──
def _enc(cfg, half):
    cfg.events.encoder_offset = EventTerm(func=mdp.randomize_encoder_offset, mode="reset",
                                          params={"offset_range": (-half, half), "asset_cfg": SceneEntityCfg("robot", joint_names=DR_LEG_JOINTS)})

def gene_enc03(cfg): _enc(cfg, 0.03)
def gene_enc05(cfg): _enc(cfg, 0.05)

def gene_com(cfg):
    cfg.events.randomize_torso_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup",
                                               params={"com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025)},
                                                       "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})

def gene_mass(cfg):
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)

def gene_fric(cfg):
    cfg.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.2)

def gene_gains(cfg):
    cfg.events.randomize_motor_strength.params["stiffness_distribution_params"] = (0.85, 1.15)
    cfg.events.randomize_motor_strength.params["damping_distribution_params"] = (0.85, 1.15)

def gene_lag(cfg):
    cfg.actions.JointPositionAction.max_lag_steps = 1

def gene_enc_com(cfg): gene_enc05(cfg); gene_com(cfg)
def gene_enc_plant(cfg): gene_enc05(cfg); gene_fric(cfg); gene_gains(cfg)
def gene_phys(cfg): gene_mass(cfg); gene_fric(cfg); gene_gains(cfg); gene_lag(cfg)
def gene_all(cfg): gene_enc05(cfg); gene_com(cfg); gene_mass(cfg); gene_fric(cfg); gene_gains(cfg); gene_lag(cfg)


def _hipsarm_dr_stack(cfg):
    _swap_in_ik_command(cfg)
    _make_hipsarm(cfg)
    _use_real_hand(cfg)
    _make_hipsarm_dr(cfg)


@configclass
class RobotEnvCfgHipsArmDR(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _hipsarm_dr_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgHipsArmDR(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _hipsarm_dr_stack(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
