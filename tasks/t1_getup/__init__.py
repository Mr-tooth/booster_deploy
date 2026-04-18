"""Task registrations for T1 getup deployment presets."""

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .t1_getup import T1GetupControllerCfg


@configclass
class T1GetupSim2SimControllerCfg(T1GetupControllerCfg):
    """T1 getup preset for MuJoCo sim2sim validation."""

    def __post_init__(self) -> None:
        """Populate default exported actor checkpoint path."""
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_getup_actor.pt"


@configclass
class T1GetupSim2RealControllerCfg(T1GetupControllerCfg):
    """T1 getup preset for sim2real runtime deployment."""

    def __post_init__(self) -> None:
        """Populate default exported actor checkpoint path."""
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_getup_actor.pt"


register_task("t1_getup_sim2sim", T1GetupSim2SimControllerCfg())
register_task("t1_getup_sim2real", T1GetupSim2RealControllerCfg())
