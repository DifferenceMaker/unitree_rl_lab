"""LM7G terms (2026-10-28): the stepping turn.

Own module (the lm7u / lm7e / lm7f convention: a new file + one appended import line in mdp/__init__.py).

Operator 2026-10-28, after the lm7f sim2sim (every row slides its feet when rotating in place; kit_gforce turns at ~0.3 of a -0.5
command "at a continuous rate" but "slides its leg when rotating"): "I raise my one leg, rotate it towards the wz i need, lower it
back down ... and then i do the same with the other leg to match the first leg. That rotation of how much the legs need to be rotated
out is basically how much the wz is. I don't want the robot to always stampede when rotating but i also don't want the robot to slide
its feet when rotating."

Why the policies slide: `feet_slide` (the trunk's -2.0) prices the LINEAR xy speed of a loaded foot. A foot pivoting about its own
centre has almost no xy speed at the ankle body origin, so the pivot is free; the swing (lift, rotate, place) costs clearance effort
and the gait phase. `feet_pivot` is the missing mirror: the YAW RATE of a loaded foot, charged like slide. In a stepping turn the
planted foot does not rotate against the ground (the hip yaw joint absorbs the body's rotation), so a correct turn pays ~0 -- a
constraint, not a cost (Bible rule 2). Gate = the same contact flag feet_slide uses (precedent in the trunk since lm3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_pivot(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum over the feet of |foot yaw rate (world z)| x in-contact (net force > 1 N over the sensor history), rad/s.

    The rotational twin of isaaclab's `feet_slide` (|foot xy speed| x in-contact). A loaded foot that turns on the spot is priced; a
    foot in the air is free to be rotated to the new heading; a planted foot under a body that yaws about it reads ~0 (its own yaw
    rate is ~0 while the hip yaw joint turns). At -2.0 a both-feet sliding turn at 0.3 rad/s pays 2 x 0.3 x 2 = 1.2/s."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    wz = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2].abs()
    return torch.sum(wz * contacts, dim=1)
