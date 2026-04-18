from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .loco_crawl_dwaq import T1CrawlDwaqControllerCfg


@configclass
class T1CrawlDwaqControllerDeployCfg(T1CrawlDwaqControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.policy.checkpoint_path = "models/Mar08_22-35-53__ite10200.pt"
        # self.policy.checkpoint_path = "models/Mar12_00-05-38__ite19400.pt"
        # self.policy.checkpoint_path = "models/Mar12_00-05-38__ite19400.onnx"
        # self.policy.checkpoint_path = "models/Mar12_13-24-32__ite2200.onnx"
        self.policy.checkpoint_path = "models/Mar12_15-53-08__ite2400.onnx"


register_task("t1_crawl_dwaq", T1CrawlDwaqControllerDeployCfg())
