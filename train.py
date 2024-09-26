import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import torch
from fvcore.nn.precise_bn import get_bn_modules
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from diffusioninst import DiffusionInstDatasetMapper, add_diffusioninst_config, DiffusionInstWithTTA
from diffusioninst.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
import logging
import os
import time
import weakref
import itertools
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict


class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionInst. """

    def __init__(self, cfg):
        self._skip_periodic_writer = False
        super().__init__(cfg)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        self.model = self.build_model(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.data_loader = self.build_train_loader(cfg)
        self._data_loader_iter = iter(self.data_loader)  # Initialize the data loader iterator

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        kwargs = {'trainer': weakref.proxy(self)}
        kwargs.update(may_get_ema_checkpointer(cfg, self.model))
        self.checkpointer = DetectionCheckpointer(self.model, cfg.OUTPUT_DIR, **kwargs)
        self.start_iter = self.checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", 0)
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.losses = []
        self._skip_periodic_writer = False  
        self.register_hooks(self.build_hooks())

        with open(os.path.join(cfg.OUTPUT_DIR, "training_info.txt"), "w") as f:
            f.write(f"Max Iterations: {self.max_iter}\n")
            f.write(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}\n")

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)  # Get the next batch from the iterator
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self.losses.append(losses.item())
        assert torch.isfinite(losses).all(), loss_dict

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self._write_metrics(loss_dict, data_time)

    def train(self, start_iter, max_iter):
        self.start_iter = start_iter
        self.max_iter = max_iter
        super().train()  # Call the superclass method
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.losses)), self.losses, label='Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Convergence')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, "loss_convergence.png"))
        plt.show()

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionInstDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionInstWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if cfg.SOLVER.CHECKPOINT_PERIOD > 0:
            ret.append(
                hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD)
            )

        if not hasattr(cfg.SOLVER, "LOG_PERIOD"):
            cfg.SOLVER.LOG_PERIOD = 20
        if not self._skip_periodic_writer:
            ret.append(
                hooks.PeriodicWriter(self.build_writers(), period=cfg.SOLVER.LOG_PERIOD)
            )
        return ret

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusioninst_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
   # cfg.MODEL.WEIGHTS = "/data1/ymliu/DiffusionInst-main/DiffusionInst-main/datasets/5-class/SEM/checkpoints/diffusioninst.pth"
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"run_{args.job_id}")
    cfg.SOLVER.MAX_ITER = args.max_iter
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    trainer = Trainer(cfg)
    return trainer.train(0, cfg.SOLVER.MAX_ITER)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--max_iter", type=int, default=30000, help="Max iterations for training"
    )
    parser.add_argument(
        "--job_id", type=int, default=0, help="Job ID for output directory naming"
    )
    args = parser.parse_args()
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
