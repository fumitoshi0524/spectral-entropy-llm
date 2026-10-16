"""Downstream evaluation via lm-evaluation-harness."""

import logging

logger = logging.getLogger(__name__)


def evaluate_downstream(model, tokenizer, tasks, num_fewshot=0):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate

    model.eval()
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

    results = {}
    for task in tasks:
        logger.info("  evaluating %s...", task)
        try:
            r = simple_evaluate(
                model=lm_obj, tasks=[task], num_fewshot=num_fewshot,
                batch_size="auto", limit=None, log_samples=False,
            )
            results[task] = {
                k: round(v, 4) for k, v in r["results"].get(task, {}).items()
                if not k.startswith("alias") and isinstance(v, (int, float))
            }
        except Exception as e:
            logger.error("  %s failed: %s", task, e)
            results[task] = {"error": str(e)}
    return results
