
from openood.utils import Config


from .base_evaluator import BaseEvaluator
from .fsood_evaluator_clip import FSOODEvaluatorClip, OODEvaluatorClip, OODEvaluatorClipTTA

def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'fsood_clip': FSOODEvaluatorClip,
        'ood_clip': OODEvaluatorClip,
        'ood_clip_tta': OODEvaluatorClipTTA
    }
    return evaluators[config.evaluator.name](config)
