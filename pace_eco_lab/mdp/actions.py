"""PACE 关节位置动作及软/硬限位保护。"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .joint_limits import safe_joint_position_targets


class PaceJointPositionAction(JointPositionAction):
    """相对默认姿态的关节目标，并在写入模拟器前应用 PACE 限位规则。"""

    cfg: "PaceJointPositionActionCfg"

    def apply_actions(self):
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        hard_limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        current = self._asset.data.joint_pos[:, self._joint_ids]
        safe_target = safe_joint_position_targets(self.processed_actions, current, soft_limits, hard_limits)
        self._asset.set_joint_position_target(safe_target, joint_ids=self._joint_ids)


@configclass
class PaceJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = PaceJointPositionAction
