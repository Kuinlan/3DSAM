import torch
import torch.nn as nn
from .attention import PerceiverLayer

class PerceiverBasicDecoder(nn.Module):
    """
    Basic decoder network for Perceiver, with functionalities shared across specific decoders
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """    
    def __init__(self, config, output_num_channels=None):
        super().__init__()

        self.decoding_cross_attention = PerceiverLayer(
            is_cross_attention=True,
            cross_attention_shape=config['cross_attention_shape'],
            use_query_residual=config['use_query_residual'],
            qk_channels=config['qk_channels'],
            v_channels=config['v_channels'],
            num_heads=config['num_heads'],
            q_dim=config['num_channels'],
            kv_dim=config['d_latents'],
            widening_factor=config['widening_factor'],
            hidden_activ=config['hidden_activ'],
            attention_dropout=config['attention_dropout'],
            use_flash_attention=config['use_flash_attention'],
        )

        self.mlp_type = config['mlp_type']
        if self.mlp_type == 'single':
            self.final_layer = nn.Linear(config['num_channels'], output_num_channels)
            # self.final_layer.weight.data.normal_(mean=0.0, std=0.02)
            # self.final_layer.bias.data.fill_(0.0)
        elif self.mlp_type == 'double':
            self.final_layer = nn.Sequential(
                nn.Linear(config['num_channels'], 2 * config['num_channels']),
                nn.Linear(2 * config['num_channels'], output_num_channels),
            )

    def forward(self, query, z, query_mask=None):
        """Forward pass of the decoder network, returns attention values and predictions"""

        cross_output = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
        )

        predictions = self.final_layer(cross_output['mlp'])

        return {
            'predictions': predictions,
            'cross_output': cross_output,
        }

class BaseDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_num_channels = config['output_num_channels']
        self.max_queries = config['max_queries']
        self.decoder = PerceiverBasicDecoder(config, config['output_num_channels'])

    def forward(self, query, z):
        # Decode queries
        s, t = self.max_queries, query.shape[1]  # query: [1, 245760, 99]
        steps = t // s + 1
        cross_outputs = []
        for i in range(0, steps):
            st, fn = s * i, min(t, s * (i + 1))
            cross_outputs.append(self.decoder(
                query[:, st:fn], z, None)
            )

        cross_output = {'predictions': torch.cat([val['predictions'] for val in cross_outputs], 1)}
        depth = {'embed': torch.cat([val['cross_output']['context'] for val in cross_outputs], 1)}
        pred = cross_output['predictions']
        depth_embed = depth['embed']

        return {
            'predictions': pred,
            'depth_embed': depth_embed,  # 1024
            'cross_output': cross_output,
        }


# class DepthDecoder(BaseDecoder):
#     """
#     Perceiver IO depth decoder

#     Parameters
#     ----------
#     cfg : Config
#         Configuration with parameters
#     """
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.output_mode = cfg.output_mode
#         self.sigmoid = torch.nn.Sigmoid()
#         if self.output_mode == 'inv_depth':
#             self.sigmoid_to_depth = SigmoidToInvDepth(
#                 min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
#         elif self.output_mode == 'inv_depth+logvar':
#             self.sigmoid_to_depth = SigmoidToInvDepth(
#                 min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
#         elif self.output_mode == 'log_depth':
#             self.sigmoid_to_log_depth = SigmoidToLogDepth()
#         elif self.output_mode == 'mixture':
#             self.sigmoid_to_depth = SigmoidToInvDepth(
#                 min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
#         elif self.output_mode == 'bins':
#             self.return_training_depth = cfg.has('return_training_depth', False)
#             self.sampling_type = cfg.has('sampling_type', 'linear')
#             self.distribution = get_depth_bins(
#                 self.sampling_type, cfg.depth_range[0], cfg.depth_range[1], self.output_num_channels)
#         else:
#             raise ValueError('Invalid depth output mode')

#     def process(self, pred, info, previous):
#         """Process the output of the decoder"""
#         if self.output_mode == 'inv_depth':
#             pred = {
#                 'depth': self.sigmoid_to_depth(self.sigmoid(pred)),
#             }
#         elif self.output_mode == 'inv_depth+logvar':
#             pred = {
#                 'depth': self.sigmoid_to_depth(self.sigmoid(pred[:, [0]])),
#                 'logvar': pred[:, [1]]
#             }
#         elif self.output_mode == 'log_depth':
#             pred = {
#                 'depth': self.sigmoid_to_log_depth(self.sigmoid(pred))
#             }
#         elif self.output_mode == 'bins':
#             b, c, h, w = pred.shape
#             pred = {
#                 'bins': pred,
#                 'zvals': self.distribution.to(pred.device),
#             }
#             if not self.training or (self.training and self.return_training_depth):
#                 idx = torch.argmax(pred['bins'], dim=1, keepdim=True)
#                 bins = self.distribution.view(1, -1, 1, 1).repeat(b, 1, h, w).to(pred['bins'].device)
#                 pred['depth'] = torch.gather(bins, 1, idx)
#         elif self.output_mode == 'mixture':
#             pred[:, [0]] = self.sigmoid_to_depth(self.sigmoid(pred[:, [0]]))
#             pred[:, [1]] = self.sigmoid_to_depth(self.sigmoid(pred[:, [1]]))
#             pred[:, [2]] = 10 * self.sigmoid(pred[:, [2]])
#             pred[:, [3]] = 10 * self.sigmoid(pred[:, [3]])
#             pred[:, [4]] = self.sigmoid(pred[:, [4]])
#         return pred