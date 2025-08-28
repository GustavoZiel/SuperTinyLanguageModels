"""Code for running samples from the evaluation benchmarks"""

from evals.load_evaluators import load_evaluator


def train_eval(eval_name, eval_cfg, model):
    """Train the model"""
    kwargs = {key: value for key, value in eval_cfg.items() if key != "evaluator"}
    evaluator = load_evaluator(eval_name, model, **kwargs)
    results = evaluator.evaluate()
    return results
