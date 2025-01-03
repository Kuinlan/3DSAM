from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from einops.einops import rearrange
class ThreeDSAMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['threedsam']['loss']
        self.match_type = self.config['threedsam']['match_coarse']['match_type']
        self.sparse_spvs = self.config['threedsam']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']

        # fine-level
        self.fine_type = self.loss_config['fine_type']

        # depth-map
        self.depth_max = self.loss_config['depth_max']
        self.depth_min = self.loss_config['depth_min']
        self.num_bins = self.loss_config['num_bin']

    def compute_depth_loss(self, depth_logits0, depth_logits1, depth_map0, depth_map1, data):
        assert 'depth0' in data, 'We need to supervise depth information during training.'
        # downsampling to coarse scale (where depth info stays at)
        gt_depth_map0 = interpolate(data['depth0'].unsqueeze(1), data['hw0_c'], mode='nearest')
        gt_depth_map1 = interpolate(data['depth1'].unsqueeze(1), data['hw1_c'], mode='nearest')
        gt_depth_map0 = rearrange(gt_depth_map0, 'n c h w -> (n c) h w')
        gt_depth_map1 = rearrange(gt_depth_map1, 'n c h w -> (n c) h w')
        
        # generate mask for background
        bg_mask0 = gt_depth_map0 < 1e-5
        bg_mask1 = gt_depth_map1 < 1e-5

        # 0. filter bg mask for ground truth depth maps
        gt_depth_map0[bg_mask0] = self.depth_max
        gt_depth_map1[bg_mask1] = self.depth_max
        
        # 1. calculate focal loss for depth logits
        bin_size = (self.depth_max - self.depth_min) / (self.num_bins - 1)
        indices0 = (gt_depth_map0 - self.depth_min) / bin_size
        indices1 = (gt_depth_map1 - self.depth_min) / bin_size
        indices0 = indices0.type(torch.int64) # [N H W]
        indices1 = indices1.type(torch.int64)

        # For image
        input_soft0 = F.softmax(depth_logits0, dim=1)
        log_input_soft0 = F.log_softmax(depth_logits0, dim=1)
        
        shape0 = indices0.shape
        target_one_hot0 = torch.zeros((shape0[0], depth_logits0.shape[1]) + shape0[1:], device=depth_logits0.device, dtype=depth_logits0.dtype)
        print(indices0.max())
        target_one_hot0 = target_one_hot0.scatter_(1, indices0.unsqueeze(1), 1.0) + 1e-6
        
        weight0 = torch.pow(-input_soft0 + 1.0, self.loss_config['focal_gamma'])
        focal0 = -self.loss_config['focal_alpha'] * weight0 * log_input_soft0
        loss_focal0 = torch.einsum('bc...,bc...->b...', (target_one_hot0, focal0))
        # loss_focal0_mean = 1.0 * loss_focal0[bg_mask0].sum() + 10.0 * loss_focal0[~bg_mask0].sum()
        # loss_focal0_mean = loss_focal0_mean / (data['hw0_c'][0] * data['hw0_c'][1])

        # For image1
        input_soft1 = F.softmax(depth_logits1, dim=1)
        log_input_soft1 = F.log_softmax(depth_logits1, dim=1)

        shape1 = indices1.shape
        target_one_hot1 = torch.zeros((shape1[0], depth_logits1.shape[1]) + shape1[1:], device=depth_logits1.device, dtype=depth_logits1.dtype)
        target_one_hot1 = target_one_hot1.scatter_(1, indices1.unsqueeze(1), 1.0) + 1e-6

        weight1 = torch.pow(-input_soft1 + 1.0, self.loss_config['focal_gamma'])
        focal1 = -self.loss_config['focal_alpha'] * weight1 * log_input_soft1
        loss_focal1 = torch.einsum('bc...,bc...->b...', (target_one_hot1, focal1))
        # loss_focal1_mean = 1.0 * loss_focal1[bg_mask1].sum() + 10.0 * loss_focal1[~bg_mask1].sum()
        # loss_focal1_mean = loss_focal1_mean / (data['hw1_c'][0] * data['hw1_c'][1])

        # 2. calculate l1 loss for depth loss
        loss_dense_depth0 = torch.abs((depth_map0 - gt_depth_map0)[~bg_mask0]).sum() / ((~bg_mask0).sum() + 1e-4)
        loss_dense_depth1 = torch.abs((depth_map1 - gt_depth_map1)[~bg_mask1]).sum() / ((~bg_mask1).sum() + 1e-4)

        return loss_focal0.mean()+loss_focal1.mean(), loss_dense_depth0+loss_dense_depth1

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data, depth_logits0=None, depth_logits1=None, depth_map0=None, depth_map1=None):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. depth-guided loss
        if depth_logits0 is not None: # only calculate when passed in
            loss_depth_focal, loss_depth_dense = self.compute_depth_loss(depth_logits0, depth_logits1, depth_map0, depth_map1, data)

            loss += loss_depth_focal * self.loss_config['depth_focal_weight']
            loss_scalars.update({"loss_depth_focal": loss_depth_focal.clone().detach().cpu()})

            loss += loss_depth_dense * self.loss_config['depth_dense_weight']
            loss_scalars.update({"loss_depth_dense": loss_depth_dense.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
