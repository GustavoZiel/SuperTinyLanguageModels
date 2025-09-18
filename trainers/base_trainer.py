"""Trainer class for training models with Next Token Prediction"""

import datetime
import time
from contextlib import nullcontext
from copy import deepcopy
from itertools import islice

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import wandb
from models import model_shell
from trainers import datasets as train_dataloader
from trainers import utils
from trainers.evaluator import train_eval
from trainers.utils import aggregate_value, print_evaluation_results
from utils.logger import get_logger

logger = get_logger(__name__)


# pylint: disable invalid-name
class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger
    """

    def __init__(
        self,
        cfg,
        model: model_shell.ModelShell,
        optimizer,
        train_dataloader,
        val_dataloader,
        loss_fn,
        gpu_id=None,
        lr_scheduler=None,
        dropout_scheduler=None,
    ) -> None:
        self.model = model
        if gpu_id is not None:  # using ddp
            self.dist = True
            self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.dist = False
            self.DDP_model = model
        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.train_dataloader_iter = iter(train_dataloader)
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.cfg = cfg
        # assert self.cfg["trainer"]["training"]["gradient_accumulation_steps"] % torch.cuda.device_count() == 0, "Gradient Accumulation Steps must be divisible by the number of GPUs"
        self.gradient_accumulation_steps = (
            cfg["trainer"]["training"]["gradient_accumulation_steps"] // torch.cuda.device_count()
            if torch.cuda.is_available()
            else cfg["trainer"]["training"]["gradient_accumulation_steps"]
        )  ## divide by number of GPUs to maximise throughput
        self.scaler = None
        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.cached_sets = {"train": {}, "val": {}}
        self.batch_size = cfg["trainer"]["training"]["batch_size"]  ## new

        # For training, always force the device to be cuda
        # assert torch.cuda.is_available(), "CUDA must be available for training"
        self.ctx = self._setup_ctx()
        if self.use_wandb and (
            self.gpu_id == 0 or not self.dist
        ):  ## ensures that only the first GPU logs to wandb
            self._setup_logging()
        if cfg.trainer.training.run_profiler and (
            self.gpu_id == 0 or not self.dist
        ):  ## ensures that only the first GPU runs the profiler
            self.run_profile()
            raise SystemExit

    def _setup_logging(self):
        # set run name
        run_name = (
            f"{self.cfg.model['model_shell_type']}"
            f"_{self.cfg.model['core_model']['core_model_type']}"
            f"_{self.cfg.trainer['dataset']}_{self.cfg.model['embedder']['embedding_model_type']}"
            f"_{self.cfg.model['vocab_size']}"
        )
        wandb.init(
            project=self.cfg.general.logging.wandb_project,
            config=OmegaConf.to_container(self.cfg),
            name=run_name,
        )
        wandb.init(project=self.cfg.general.logging.wandb_project)
        print("wand_b_initted")

    def _setup_ctx(self):
        """Get the context manager"""
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._setup_scaler(dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        return ctx

    def _setup_scaler(self, dtype=torch.float16):
        """Setup the scaler"""
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=dtype == torch.float16)

    @torch.no_grad()
    def estimate_performance(
        self, eval_iters: int = None, verbose: bool = False
    ) -> tuple[dict, dict]:
        """Estimate the loss and perplexity on the validation set, plus evaluator metrics.

        Args:
            eval_iters (int, optional): Number of evaluation iterations. Defaults to config value.
            verbose (bool, optional): If True, logs detailed info. Defaults to False.

        Returns:
            tuple[dict, dict]: (eval_results, evaluator_results)
        """
        if verbose:
            logger.info("estimate_performance: called")
        if eval_iters is None:
            eval_iters = self.cfg.trainer.training.eval_iters
        if verbose:
            logger.info(f"estimate_performance: eval_iters={eval_iters}")
        eval_results: dict = {}
        self.model.eval()

        # eval on val set
        losses = []
        perplexities = []
        for i, (x, y) in enumerate(self.val_dataloader):
            if verbose:
                logger.info(f"estimate_performance: batch {i}")
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            with self.ctx:
                output, _ = self.model(x)
                loss = self.loss_fn(output, y)
                if verbose:
                    logger.info(f"estimate_performance: loss={loss.item()}")
                losses.append(loss.item())
                perplexity = torch.exp(loss)
                if verbose:
                    logger.info(f"estimate_performance: perplexity={perplexity.item()}")
                perplexities.append(perplexity.item())
            if i >= eval_iters:
                if verbose:
                    logger.info("estimate_performance: reached eval_iters limit, breaking")
                break

        avg_loss = aggregate_value(np.mean(losses), self.cfg.general.device)
        if verbose:
            logger.info(f"estimate_performance: avg_loss={avg_loss}")
        eval_results["Loss"] = avg_loss

        avg_perplexity = aggregate_value(np.mean(perplexities), self.cfg.general.device)
        if verbose:
            logger.info(f"estimate_performance: avg_perplexity={avg_perplexity}")
        eval_results["Perplexity"] = avg_perplexity

        evaluator_results: dict = {}
        for evaluator in self.cfg.trainer["eval"]["evaluator"]:
            if verbose:
                logger.info(f"estimate_performance: running evaluator {evaluator}")
            evaluator_metrics = train_eval(
                eval_name=evaluator, eval_cfg=self.cfg.trainer["eval"], model=self.model
            )
            relabeled_results = {
                f"{evaluator}/{metric}": value for metric, value in evaluator_metrics.items()
            }
            evaluator_results[evaluator] = relabeled_results
        self.model.train()
        if verbose:
            logger.info("estimate_performance: returning results")
        return eval_results, evaluator_results

    def _run_step(self):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        accumulated_loss = 0
        for i in range(self.gradient_accumulation_steps):
            # get the next batch
            x, y = next(self.train_dataloader_iter)
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)

            # Enable or disable gradient synchronization based on the need for accumulation
            if self.dist and hasattr(self.DDP_model, "no_sync"):
                context_manager = (
                    self.DDP_model.no_sync()
                    if i != self.gradient_accumulation_steps - 1
                    else nullcontext()
                )
            else:
                context_manager = nullcontext()

            with context_manager:
                with self.ctx:
                    output, aux_loss = self.DDP_model(x)
                    loss = self.loss_fn(output, y)
                    if aux_loss is not None:
                        loss += aux_loss

                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

        # once graidents are accumulated, step
        if self.cfg.trainer.optimizer.grad_clip > 0:
            # Unscale the gradients of the optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip the gradients with normalization
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.trainer.optimizer.grad_clip
            )

        # Perform a single optimization step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()  # Reset gradients after update

        return accumulated_loss

    def run_profile(self):
        """Run the profiler"""
        utils.profilize(self.model)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(10):
                if i <= 3:
                    self._run_step()  ## set the 'epoch' to ensure shuffle
                else:
                    with record_function("_run_step"):
                        self._run_step()  ## set the 'epoch' to ensure shuffle
            # place profile in dictionary
        backwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(backwards_prof)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self.estimate_performance(eval_iters=1)
            with record_function("estimate_performance"):
                self.estimate_performance(eval_iters=10)
            # place profile in dictionary
        forwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(forwards_prof)

    def save_checkpoint(self, iteration: int, verbose: bool = True) -> None:
        """Save the current model checkpoint to disk.

        Args:
            iteration (int): The current training iteration number.
            verbose (bool, optional): If True, logs checkpoint saving info. Defaults to True.

        The checkpoint includes:
            - Model state dictionary
            - Optimizer state dictionary
            - Current iteration number
            - Configuration object
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": iteration,
            "config": self.cfg,
        }
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_path = (
            f"{self.checkpoint_dir}/{current_time}_{self.cfg.trainer.dataset}_{iteration}.pt"
        )
        if verbose:
            logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def run_training_loop(self):
        """Run the main training loop for the model.

        This method handles the following:
            - Iterates for the configured number of training steps.
            - Adjusts learning rate and dropout via schedulers.
            - Periodically evaluates model performance and logs results.
            - Saves checkpoints at specified intervals.
            - Logs metrics to wandb if enabled.
        """
        elapsed_time = 0.0
        for iter_num in range(self.cfg.trainer.training.max_iters):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num)

            # Periodic evaluation
            if self.cfg.trainer.training.eval_interval > 0 and (
                not iter_num % self.cfg.trainer.training.eval_interval
            ):
                eval_results, benchmark_results = self.estimate_performance(verbose=False)
                print_evaluation_results(
                    iter_num=iter_num,
                    eval_results=eval_results,
                    benchmark_results=benchmark_results,
                )
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    log_dict = {"iter": iter_num, "lr": lr, "dropout": dropout}
                    log_dict.update(eval_results)
                    log_dict.update({k: v for k, v in benchmark_results.items()})
                    print("Wand db Log dict keys:", log_dict.keys())
                    wandb.log(log_dict)

            # Periodic checkpointing
            if (
                not iter_num % self.cfg.trainer.training.checkpoint_interval
                and iter_num > 0
                and (self.gpu_id == 0 or self.gpu_id is None)
            ):
                self.save_checkpoint(iter_num)

            # Training step
            lossf = self._run_step()
            end_time = time.time()
            elapsed_time += end_time - start_time

            # Periodic logging
            if not iter_num % self.cfg.trainer.training.log_interval and iter_num > 0:
                lossf = aggregate_value(lossf, self.cfg.general.device)
                elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                logger.info(
                    f"All GPU(s): Step {iter_num} | Loss: {lossf:.4f} | LR: {lr:.1e} | Dropout: {dropout:.2f} | Step time: {end_time - start_time:.2f}s | Total time: {elapsed_time_str}"
                )
                if (self.gpu_id == 0 or self.gpu_id is None) and self.use_wandb:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "loss": lossf,
                            "lr": lr,
                            "dropout": dropout,
                        }
                    )
        # Save the final model checkpoint
        if self.gpu_id == 0 or self.gpu_id is None:
            self.save_checkpoint(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
