"""Trainer class for training models with Next Token Prediction"""

import datetime
import time
from ast import List
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
from models.generator import StandardGenerator
from trainers import datasets as train_dataloader
from trainers import utils
from trainers.evaluator import train_eval
from trainers.utils import aggregate_value, print_evaluation_results
from utils.logger import get_logger
from wandb import Table

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
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler

        self.train_dataloader_iter = iter(train_dataloader)
        self.val_dataloader = val_dataloader

        self.cfg = cfg

        # assert self.cfg["trainer"]["training"]["gradient_accumulation_steps"] % torch.cuda.device_count() == 0, "Gradient Accumulation Steps must be divisible by the number of GPUs"
        self.gradient_accumulation_steps = (
            cfg["trainer"]["training"]["gradient_accumulation_steps"] // torch.cuda.device_count()
            if torch.cuda.is_available()
            else cfg["trainer"]["training"]["gradient_accumulation_steps"]
        )  ## divide by number of GPUs to maximise throughput

        self.iter_start = 1
        self.scaler = None
        self.batch_size = cfg["trainer"]["training"]["batch_size"]  ## new

        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.cached_sets = {"train": {}, "val": {}}
        self.table = None

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
        if cfg.trainer.training.prompt_interval > 0:
            self.table = wandb.Table(columns=["iteration", "text"], log_mode="MUTABLE")

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
        for evaluator_cfg in self.cfg.trainer["eval"]:
            if verbose:
                logger.info(f"estimate_performance: running evaluator {evaluator_cfg['evaluator']}")
            evaluator_results[evaluator_cfg["evaluator"]] = train_eval(evaluator_cfg, self.model)
            relabeled_results = {}
            for metric in evaluator_results[evaluator_cfg["evaluator"]]:
                relabeled_results[f"{evaluator_cfg['evaluator']}/{metric}"] = evaluator_results[
                    evaluator_cfg["evaluator"]
                ][metric]
            evaluator_results[evaluator_cfg["evaluator"]] = relabeled_results
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

    # def save_checkpoint(self, iteration: int, verbose: bool = True) -> None:
    #     """Save the current model checkpoint to disk.

    #     Args:
    #         iteration (int): The current training iteration number.
    #         verbose (bool, optional): If True, logs checkpoint saving info. Defaults to True.

    #     The checkpoint includes:
    #         - Model state dictionary
    #         - Optimizer state dictionary
    #         - Current iteration number
    #         - Configuration object
    #     """
    #     checkpoint = {
    #         "model": self.model.state_dict(),
    #         "optimizer": self.optimizer.state_dict(),
    #         "iteration": iteration,
    #         "config": self.cfg,
    #     }
    #     # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    #     checkpoint_path = (
    #         f"{self.checkpoint_dir}/{current_time}_{self.cfg.trainer.dataset}_{iteration}.pt"
    #     )
    #     if verbose:
    #         logger.info(f"Saving checkpoint to {checkpoint_path}")
    #     torch.save(checkpoint, checkpoint_path)

    def save_checkpoint(self, iteration: int, verbose: bool = True) -> None:
        """Save a comprehensive checkpoint for resuming training.

        Args:
            iteration (int): The current training iteration number.
            verbose (bool, optional): If True, logs checkpoint saving info. Defaults to True.
        """
        checkpoint = {
            # Model state
            "model": self.model.state_dict(),
            # Optimizer state (includes momentum, learning rate history, etc.)
            "optimizer": self.optimizer.state_dict(),
            # Schedulers state
            "lr_scheduler": self._get_scheduler_state(self.lr_scheduler),
            "dropout_scheduler": self._get_scheduler_state(self.dropout_scheduler),
            # Training progress
            "iteration": iteration,
            "iter_start": self.iter_start,
            # Random states for reproducibility
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": torch.random.get_state()
            if hasattr(torch.random, "get_state")
            else None,
            # CUDA random state if available
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            # Scaler state for mixed precision training
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            # Configuration
            "config": self.cfg,
            # Training metrics/cache if needed
            "cached_sets": self.cached_sets,
            # Dataloader state (to resume from correct position)
            "dataloader_state": {
                "epoch": getattr(self.train_dataloader_iter, "_epoch", 0),
                "batch_idx": getattr(self.train_dataloader_iter, "_batch_idx", 0),
            },
        }

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_path = (
            f"{self.checkpoint_dir}/{current_time}_{self.cfg.trainer.dataset}_{iteration}.pt"
        )

        if verbose:
            logger.info(f"Saving comprehensive checkpoint to {checkpoint_path}")

        torch.save(checkpoint, checkpoint_path)

        if verbose:
            logger.info(f"Checkpoint saved successfully at iteration {iteration}")

    def _get_scheduler_state(self, scheduler):
        """Get the state of a scheduler for checkpointing.

        Args:
            scheduler: The scheduler object to get state from

        Returns:
            dict: Scheduler state dictionary, or None if scheduler is None
        """
        if scheduler is None:
            return None

        # Try to get state_dict if available (for PyTorch schedulers)
        if hasattr(scheduler, "state_dict"):
            return scheduler.state_dict()

        # For custom schedulers, save their instance variables
        state = {}
        for attr_name in dir(scheduler):
            if not attr_name.startswith("_") and not callable(getattr(scheduler, attr_name)):
                try:
                    attr_value = getattr(scheduler, attr_name)
                    # Only save simple types that can be serialized
                    if isinstance(attr_value, (int, float, str, bool, list, tuple, dict)):
                        state[attr_name] = attr_value
                except Exception:
                    # Skip attributes that can't be accessed or serialized
                    continue

        return state

    def _load_scheduler_state(self, scheduler, state):
        """Load state into a scheduler from checkpoint.

        Args:
            scheduler: The scheduler object to load state into
            state: The state dictionary to load
        """
        if scheduler is None or state is None:
            return

        # Try to use load_state_dict if available (for PyTorch schedulers)
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state)
            return

        # For custom schedulers, restore their instance variables
        for attr_name, attr_value in state.items():
            if hasattr(scheduler, attr_name):
                try:
                    setattr(scheduler, attr_name, attr_value)
                except Exception:
                    # Skip attributes that can't be set
                    continue

    def load_checkpoint(self, checkpoint_path: str, verbose: bool = True) -> int:
        """Load a comprehensive checkpoint for resuming training.

        Args:
            checkpoint_path (str): Path to the checkpoint file to load.
            verbose (bool, optional): If True, logs checkpoint loading info. Defaults to True.

        Returns:
            int: The iteration number from the loaded checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            KeyError: If required keys are missing from the checkpoint.
        """
        if verbose:
            logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        # checkpoint = torch.load(
        #     checkpoint_path, map_location=self.cfg.general.device, weights_only=False
        # )

        # Validate checkpoint structure
        required_keys = ["model", "optimizer", "iteration", "config"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Missing required keys in checkpoint: {missing_keys}")

        # Load model state
        self.model.load_state_dict(checkpoint["model"])
        if verbose:
            logger.info("Model state loaded from checkpoint")

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if verbose:
            logger.info("Optimizer state loaded from checkpoint")

        # Load scheduler states
        if checkpoint.get("lr_scheduler") is not None and self.lr_scheduler is not None:
            self._load_scheduler_state(self.lr_scheduler, checkpoint["lr_scheduler"])
            if verbose:
                logger.info("LR scheduler state loaded from checkpoint")

        if checkpoint.get("dropout_scheduler") is not None and self.dropout_scheduler is not None:
            self._load_scheduler_state(self.dropout_scheduler, checkpoint["dropout_scheduler"])
            if verbose:
                logger.info("Dropout scheduler state loaded from checkpoint")

        # Load scaler state if available
        if checkpoint.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
            if verbose:
                logger.info("Scaler state loaded from checkpoint")

        # Restore random states for reproducibility
        if "torch_rng_state" in checkpoint:
            try:
                torch.set_rng_state(checkpoint["torch_rng_state"])
                if verbose:
                    logger.info("Torch RNG state restored")
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to restore Torch RNG state: {e}")

        if "numpy_rng_state" in checkpoint:
            try:
                np.random.set_state(checkpoint["numpy_rng_state"])
                if verbose:
                    logger.info("NumPy RNG state restored")
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to restore NumPy RNG state: {e}")

        if "python_rng_state" in checkpoint and hasattr(torch.random, "set_state"):
            try:
                torch.random.set_state(checkpoint["python_rng_state"])
                if verbose:
                    logger.info("Python RNG state restored")
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to restore Python RNG state: {e}")

        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
                if verbose:
                    logger.info("CUDA RNG state restored")
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to restore CUDA RNG state: {e}")
                    logger.info("Continuing without CUDA RNG state restoration")

        # Restore training progress
        iteration = checkpoint["iteration"]
        self.iter_start = checkpoint.get("iter_start", iteration)
        if verbose:
            logger.info(
                f"Training progress restored: iteration {iteration}, iter_start {self.iter_start}"
            )

        # Restore cached sets if available
        if "cached_sets" in checkpoint:
            self.cached_sets = checkpoint["cached_sets"]
            if verbose:
                logger.info("Cached sets restored")

        # Restore dataloader state if available
        if "dataloader_state" in checkpoint:
            dataloader_state = checkpoint["dataloader_state"]
            if hasattr(self.train_dataloader_iter, "_epoch"):
                self.train_dataloader_iter._epoch = dataloader_state.get("epoch", 0)
            if hasattr(self.train_dataloader_iter, "_batch_idx"):
                self.train_dataloader_iter._batch_idx = dataloader_state.get("batch_idx", 0)
            if verbose:
                logger.info("Dataloader state restored")

        if verbose:
            logger.info(f"Checkpoint loaded successfully from iteration {iteration}")

        return iteration

    # def run_prompting(self, prompt_cfg) -> Table:
    #     """Generate answers for a set of prompts using the model and log them in a wandb Table.

    #     Args:
    #         prompt_cfg (dict): Configuration containing 'generator' settings and 'input_prompts' list.

    #     Returns:
    #         Table: A wandb Table containing prompts and their generated answers.
    #     """
    #     generator = StandardGenerator(model=self.model, generate_cfg=prompt_cfg["generator"])
    #     log_buffer = []
    #     for input_prompt in prompt_cfg["input_prompts"]:
    #         generated_text = generator.default_generate(input_text=input_prompt)
    #         logger.info(f"Prompt: {input_prompt}\nGenerated: {generated_text}")
    #         log_buffer.append((input_prompt, generated_text[0]))
    #     return log_buffer

    def run_prompting_table(self, prompt_cfg) -> Table:
        """Generate answers for a set of prompts using the model and log them in a wandb Table.

        Args:
            prompt_cfg (dict): Configuration containing 'generator' settings and 'input_prompts' list.

        Returns:
            Table: A wandb Table containing prompts and their generated answers.
        """
        generator = StandardGenerator(model=self.model, generate_cfg=prompt_cfg["generator"])
        generated = ""
        for input_num, input_prompt in enumerate(prompt_cfg["input_prompts"], start=1):
            generated_text = generator.default_generate(input_text=input_prompt)
            generated += (
                "=" * 30 + f"\n\nQuestion {input_num}\n\n"
                f"Prompt:\n{input_prompt}\n\n"
                f"Generated:\n{generated_text[0]}\n\n"
            )
        return generated

    def run_training_loop(self, verbose: bool = True):
        """Run the main training loop for the model.

        This method handles the following:
            - Iterates for the configured number of training steps.
            - Adjusts learning rate and dropout via schedulers.
            - Periodically evaluates model performance and logs results.
            - Saves checkpoints at specified intervals.
            - Logs metrics to wandb if enabled.
        """
        elapsed_time = 0.0
        # Start from iter_start if resuming from checkpoint, otherwise start from 1
        start_iter = max(1, self.iter_start)
        for iter_num in range(start_iter, self.cfg.trainer.training.max_iters + 1):
            start_time = time.time()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num - 1)
            else:
                lr = self.optimizer.param_groups[0]["lr"]
            dropout = self.dropout_scheduler.step(self.model, iter_num - 1)

            # Periodic prompting
            if self.use_wandb and (
                iter_num == self.iter_start
                or (
                    self.cfg.trainer.training.prompt_interval > 0
                    and (not iter_num % self.cfg.trainer.training.prompt_interval)
                )
            ):
                if verbose:
                    logger.info(f"Running prompting at iteration {iter_num}")
                generated = self.run_prompting_table(self.cfg.trainer.prompt)
                self.table.add_data(iter_num, generated)
                wandb.log({"prompt_answer_table": self.table})
                # artifact_name = f"prompts_{iter_num}"
                # artifact = wandb.Artifact(name=artifact_name, type="model_predictions")
                # artifact.add(table, "predictions_table")
                # wandb.log_artifact(artifact)

            # Periodic evaluation
            if iter_num == self.iter_start or (
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
            if iter_num == self.iter_start or (
                not iter_num % self.cfg.trainer.training.checkpoint_interval
                and (self.gpu_id == 0 or self.gpu_id is None)
            ):
                self.save_checkpoint(iter_num)

            # Training step
            lossf = self._run_step()
            end_time = time.time()
            elapsed_time += end_time - start_time

            # Periodic logging
            if iter_num == self.iter_start or (
                not iter_num % self.cfg.trainer.training.log_interval
            ):
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
        # # Save the final model checkpoint
        # if self.gpu_id == 0 or self.gpu_id is None:
        #     self.save_checkpoint(iter_num)

    def train(self, seed=42):
        """Train the model"""
        utils.set_seed(seed)
        self.run_training_loop()
