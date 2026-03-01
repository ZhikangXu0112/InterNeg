from openood.utils import Config


from .ttaprompt_postprocessor import TTAPromptPostprocessor, TTAPromptLocalfeatPostprocessor, TTAPromptPostprocessor_noadagap
from .oneoodprompt_postprocessor import OneOodPromptPostprocessor, OneOodPromptDevelopPostprocessor
from .mcm_postprocessor import MCMPostprocessor



def get_postprocessor(config: Config):
    postprocessors = {
        'mcm': MCMPostprocessor,
        'oneoodprompt':OneOodPromptPostprocessor,
        'oneoodpromptdevelop':OneOodPromptDevelopPostprocessor,
        'ttaprompt': TTAPromptPostprocessor,
        'ttapromptnoadagap': TTAPromptPostprocessor_noadagap,
    }

    return postprocessors[config.postprocessor.name](config)
