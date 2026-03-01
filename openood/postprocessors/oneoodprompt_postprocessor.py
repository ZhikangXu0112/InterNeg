from typing import Any
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from openood.networks.clip import clip
import pdb

# the last output dim is ood dim.
class OneOodPromptPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = self.args.beta
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score  # sum | max
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
    
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # id_loader_dict['train']
        pass
        # pdb.set_trace()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        # class_num = 1000
        image_features, text_features, logit_scale = net(data, return_feat=True)
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention,
        if len(text_features.shape) == 3: ## 11K*7*512, weighting the text features instead of simple average, Do not work, not used. 
            sim = text_features @ image_features.t()  ## 11k*7*256
             #### may combine with temperature and softmax !!, here use cose sim directly. here with negative values.
            sim = torch.exp(-self.beta * (-sim + 1))
            temp = sim.unsqueeze(0).transpose(0,-1) * text_features.unsqueeze(0) ## 256*11k*7*1 * 1*11K*7*512 -->256, 11k, 7, 512
            sa_text_features = temp.sum(2) ## 256*11k*512
            sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm.
            output = (image_features.unsqueeze(1) * sa_text_features).sum(-1) ## 256*11k
        else:
            output = logit_scale * image_features @ text_features.t() # batch * class.
        
        _, pred_in = torch.max(output[:, :class_num], dim=1)

        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in_vanilla = scores.mean(dim=-1) ### the mean ID score of multiple groups. 
        conf = conf_in_vanilla
        
        # ############################### only score in.
        # output_only_in = output[:, :class_num]
        # output_only_out = output[:, class_num:]
        # score_only_in = torch.softmax(output_only_in / self.tau, dim=1)
        # conf_only_in, pred_only_in = torch.max(score_only_in, dim=1)
        # cosin_only_in, _ = torch.max(output_only_in, dim=1)
        # ############################## including score out. 
        # score = torch.softmax(output / self.tau, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)
        # if self.in_score == 'oodscore' or self.in_score == 'sum':
        #     conf = conf_in 
        ################# tested variants, not effective.
        # elif self.in_score == 'oodscore_wiwopt_sum':
        #     vanilla_text_classifier = net.text_features.mean(1)
        #     vanilla_text_classifier /= vanilla_text_classifier.norm(dim=-1, keepdim=True)  ## renorm.
        #     output_vanilla = logit_scale * image_features @ vanilla_text_classifier.t() # batch * class.
        #     score_vanilla = torch.softmax(output_vanilla, dim=1)
        #     conf_in_pt = torch.sum(score_vanilla[:, :class_num], dim=1)
        #     # pdb.set_trace()
        #     # # self.text_features
        #     conf = conf_in + conf_in_pt  
        # elif self.in_score == 'oodscore_wiwopt_mul':
        #     vanilla_text_classifier = net.text_features.mean(1)
        #     vanilla_text_classifier /= vanilla_text_classifier.norm(dim=-1, keepdim=True)  ## renorm.
        #     output_vanilla = logit_scale * image_features @ vanilla_text_classifier.t() # batch * class.
        #     score_vanilla = torch.softmax(output_vanilla, dim=1)
        #     conf_in_pt = torch.sum(score_vanilla[:, :class_num], dim=1)
        #     # pdb.set_trace()
        #     # # self.text_features
        #     conf = conf_in * conf_in_pt 
        # elif self.in_score == 'oodscore_cosin':
        #     # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
        #     conf = conf_out * cosin_only_in  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        # elif self.in_score == 'oodscore_cosout':
        #     # conf = conf * conf_only_in  # 这个比conf 自己还差，加入msp 结果变差了。。
        #     # pdb.set_trace()
        #     conf = - conf_out * output_only_out[:,0]  # with softmax: relative distance;  w/o softmax: direct distance. near ood 变差，far ood 变好
        # elif self.in_score == 'maxidcosdis':
        #     conf = cosin_only_in
        # elif self.in_score == 'maxoodcosdis':
        #     conf = - output_only_out[:,0]
        # elif self.in_score == 'maxidscore':
        #     conf = conf_only_in
        # elif self.in_score == 'energy': 
        #     # bad results. 
        #     conf = self.tau * torch.log(torch.exp(output_only_in / self.tau).sum(1)) - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        # elif self.in_score == 'ood_energy': 
        #     # bad results. 
        #     conf = - self.tau * torch.log(torch.exp(output_only_out / self.tau).sum(1))
        # else:
        #     raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()
        # pdb.set_trace()
        # conf, pred = torch.max(score, dim=1)
        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau


def pca(X, k=300):
    # 中心化数据
    X_mean = X.mean(0)
    X = X - X_mean.expand_as(X)
    # SVD
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k]
    # return torch.mm(X, U[:, :k])


def encode_with_pseudo_tokens(clip_model, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1).int().cuda(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x


class OneOodPromptDevelopPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OneOodPromptDevelopPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau
        self.beta = int(self.args.beta)
        self.thres = self.args.thres
        self.extra_text_max_length= self.args.extra_text_length
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.in_score = self.args.in_score # sum | max
        self.setup_flag = False
        self.proj_flag = False
        self.group_num = self.args.group_num
        self.random_permute = self.args.random_permute
        self.extra_text_features = None 
        self.reset = True ## reset after each dataset. 
        self.bank = torch.zeros((0, 512)).half() 
        self.bank_scores = torch.zeros(0).half()  
    

    def reset_memory(self):
        self.reset = True

    # @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        foot_path = './data/txtfiles_output/'
        wordnet_processed_path = foot_path + net.backbone + '_wordnet_' + net.id_dataset + '_cossim_dedup.pth'
        wordnet_dict = torch.load(wordnet_processed_path)
        can_cos_id = wordnet_dict['cos_sim_id'] 
        class_num = net.n_cls
        class_img = net.class_img # class * feature_dim
        
        if self.reset:
            self.extra_text_features = None
            self.reset = False
            self.bank = torch.zeros((0, net.embed_dim)).half() 
            self.bank_scores = torch.zeros(0).half()  
            
        with torch.no_grad():
            net.eval()
            image_features, text_features, logit_scale = net(data, return_feat=True)

            if self.extra_text_features != None:
                output_vanilla = logit_scale * image_features @ self.extra_text_features.cuda().t()  # batch * class.
            else:
                output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class.

            _, pred_in = torch.max(output_vanilla[:, :class_num], dim=1)

            pos_logit = output_vanilla[:, :class_num] ## B*C
            neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
            drop = neg_logit.size(1) % self.group_num
            if drop > 0:
                neg_logit = neg_logit[:, :-drop]

            if self.random_permute:
                # print('use random permute')
                SEED=0
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
                neg_logit = neg_logit.T ## total_neg_num*B
                # pdb.set_trace()
                neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
            else:
                neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
            scores = []
            for i in range(self.group_num):
                full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
                full_sim = full_sim.softmax(dim=-1)
                pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
                scores.append(pos_score.unsqueeze(-1))
            scores = torch.cat(scores, dim=-1)
            conf_in = scores.mean(dim=-1)
            # return pred_in, output_vanilla.softmax(dim=-1)  # for case study

        activate_indicator = conf_in < self.thres 
        if  torch.any(activate_indicator):
            embedding_dim = net.word_embed_dim  # for word embedding
            criterion = nn.CosineEmbeddingLoss()
            criterion_target = torch.as_tensor([1], dtype=torch.float).cuda()
            templates = net.templates
            batch_im_features = image_features[activate_indicator]
            # print(batch_im_features.shape)
            bs = len(batch_im_features)
            oti_pseudo_tokens = torch.empty((bs, embedding_dim)).cuda()
            nn.init.normal_(oti_pseudo_tokens, std=0.02)
            oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)
            ema_oti_pseudo_tokens = oti_pseudo_tokens.clone().detach().cpu()
            optimizer = torch.optim.AdamW([oti_pseudo_tokens], lr=2e-2, weight_decay=0.01)
            scaler = torch.cuda.amp.GradScaler()

            for _ in range(10): 
                optimizer.zero_grad()
                template_indexes = random.choices(range(len(templates)), k=bs)
                template_oti_texts = [templates[i].format(" $ ") for i in template_indexes]
                tokenized_template_oti_texts = clip.tokenize(template_oti_texts, truncate=True).cuda()  # tokenize
                
                with torch.cuda.amp.autocast():
                    template_oti_features = encode_with_pseudo_tokens(net.model, tokenized_template_oti_texts,
                                                                    oti_pseudo_tokens)                                               
                    cosine_loss = criterion(template_oti_features.cuda(), batch_im_features, criterion_target)
                    oti_loss = cosine_loss
            
                # Backpropagate the loss
                scaler.scale(oti_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            

            with torch.no_grad():
                template_oti_features = encode_with_pseudo_tokens(net.model, tokenized_template_oti_texts,
                                                                        oti_pseudo_tokens)
                template_oti_features /= template_oti_features.norm(dim=-1, keepdim=True)    

                ##### Inter-based extra negative text ######
                new_text_features = (template_oti_features  @  class_img.T).detach().cpu()
                mask = torch.all(new_text_features < can_cos_id, dim=1)
                if mask.numel() != 0 :
                    selected_new_text_features = template_oti_features[mask].detach().cpu()
                    tmp_feature_score = (can_cos_id - new_text_features[mask]).mean(dim=1)
                  
                    combined_features = torch.cat([self.bank, selected_new_text_features], dim=0)
                    combined_scores = torch.cat([self.bank_scores, tmp_feature_score], dim=0)

             
                    sorted_scores, sorted_indices = torch.sort(combined_scores, descending=True)
                    sorted_features = combined_features[sorted_indices]

                    self.bank = sorted_features.half()
                    self.bank_scores = sorted_scores.half()
                    self.extra_text_features = torch.cat((text_features.detach().cpu(),self.bank[:self.extra_text_max_length]),dim=0)
                    # print(self.extra_text_features.shape)

                    output = logit_scale * image_features @ self.extra_text_features.cuda().t()  # batch * class.
                    _, pred_in = torch.max(output[:, :class_num], dim=1)

                    pos_logit = output[:, :class_num] ## B*C
                    neg_logit = output[:, class_num:] ## B*total_neg_num
                    drop = neg_logit.size(1) % self.group_num
                    if drop > 0:
                        neg_logit = neg_logit[:, :-drop]
                    # print(neg_logit.shape)

                    if self.random_permute:
                        # print('use random permute')
                        SEED=0
                        torch.manual_seed(SEED)
                        torch.cuda.manual_seed(SEED)
                        idx = torch.randperm(neg_logit.shape[1]).to(output.device)
                        neg_logit = neg_logit.T ## total_neg_num*B
                        # pdb.set_trace()
                        neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
                    else:
                        neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
                    scores = []
                    for i in range(self.group_num):
                        full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
                        full_sim = full_sim.softmax(dim=-1)
                        pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
                        scores.append(pos_score.unsqueeze(-1))
                    scores = torch.cat(scores, dim=-1)
                    conf_in = scores.mean(dim=-1)


        # max in prob - max out prob
        if self.in_score == 'oodscore' or self.in_score == 'sum':
            conf = conf_in  ## = 1-conf_out
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau

