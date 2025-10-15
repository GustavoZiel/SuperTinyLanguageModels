"""A collection of optimizers used for training."""

import inspect

import torch

from utils.logger import get_logger

logger = get_logger(__name__)


# pylint: disable=invalid-name
def configure_nanoGPT_optimizer(model, weight_decay, learning_rate, betas):
    """Configure the optimizer for NanoGPT"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"Num non-decayed parameter tensors: {len(nodecay_params)},"
        f" with {num_nodecay_params:,} parameters"
    )
    # Check if fused AdamW is available and use it only if CUDA is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = {"fused": use_fused} if use_fused else {}

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"Using fused AdamW: {use_fused}")

    return optimizer
