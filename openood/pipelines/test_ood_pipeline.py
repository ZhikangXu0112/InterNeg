import time

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import random

class TestOODPipeline:
    def __init__(self, config) -> None:
        self.config = config
    
    def fix_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)
        self.fix_seed(self.config.seed)

        # get dataloader
        id_loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        # init network
        net = get_network(self.config.network,id_loader_dict,ood_loader_dict)
        
        # init ood evaluator
        evaluator = get_evaluator(self.config)

        # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, id_loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # start evaluating ood detection methods
        timer = time.time()
        if self.config.evaluator.ood_scheme == 'fsood':
            evaluator.eval_ood(net,
                               id_loader_dict,
                               ood_loader_dict,
                               postprocessor,
                               fsood=True)
        else:
            evaluator.eval_ood(net, id_loader_dict, ood_loader_dict,
                               postprocessor)
        print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
        print('Completed!', flush=True)
