# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from mmcls.apis import init_model

import openood.utils.comm as comm
import pdb


from .clip_fixed import FixedCLIP
from .clip_fixed_ood_prompt import FixedCLIP_OODPrompt, FixedCLIP_NegOODPrompt, FixedCLIP_InterNegOODPrompt


def check_size_mismatches(net, checkpoint_path):
    # 加载检查点的状态字典
    checkpoint_dict = torch.load(checkpoint_path)

    # 遍历模型的参数
    for name, param in net.state_dict().items():
        if name in checkpoint_dict:
            checkpoint_param = checkpoint_dict[name]
            # 检查大小是否匹配
            if param.size() != checkpoint_param.size():
                print(f"Size mismatch for {name}: model {param.size()}, checkpoint {checkpoint_param.size()}")
        else:
            print(f"{name} is missing in checkpoint.")

def get_network(network_config,id_loader_dict=None,ood_loader_dict=None):

    num_classes = network_config.num_classes
    
    
    if network_config.name == 'fixedclip':
        net = FixedCLIP(network_config)

    elif network_config.name == 'fixedclip_oodprompt':
        net = FixedCLIP_OODPrompt(network_config)

    elif network_config.name == 'fixedclip_negoodprompt':
        net = FixedCLIP_NegOODPrompt(network_config)
    
    elif network_config.name == 'fixedclip_inter_negoodprompt':
        net = FixedCLIP_InterNegOODPrompt(network_config,id_loader_dict,ood_loader_dict)



    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet.cuda(),
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net
