#
# Edited by: Jingwei Xu, ShanghaiTech University
# Adapted from nr3d_lib and tinycudann
#

import torch
import torch.nn as nn
import torch.optim as optim
import tinycudann as tcnn

from utils.sh_encoder_utils import SHEncoder
from utils.mlp_utils import FCBlock

class NeRFEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 2^{ [0,L-1] }
            # log sampling
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # linear sampling
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)


        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,inputs):
        # -1 means concat on the last dim
        return torch.cat( [fn(inputs) for fn in self.embed_fns ], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input' :   True,
        'input_dims'    :   3,
        'max_freq_log2' :   multires-1,
        'num_freqs'     :   multires,
        'log_sampling'  :   True,
        'periodic_fns'  :   [torch.sin, torch.cos],
    }

    embedder_obj = NeRFEmbedder(**embed_kwargs)
    embed = lambda x , eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class SkyModel(nn.Module):
    def __init__(
        self,
        dir_embed_cfg:dict={'degree': 4},
        D=3, W=64, skips=[], activation='relu', output_activation='sigmoid',
        num_levels: int = 16, features_per_level: int = 2,
        weight_norm=False, dtype=torch.float, device=torch.device('cuda')
    ):
        super(SkyModel, self).__init__()
        self.embed_fn_view = SHEncoder(3, **dir_embed_cfg)
        input_ch_views = self.embed_fn_view.out_features

        self.position_embed, position_embed_ch = get_embedder(10, 0)

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'Grid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': 16,
                'base_resolution': 16,
                'include_static': False
            }
        )

        self.blocks = FCBlock(
            input_ch_views + num_levels * features_per_level + position_embed_ch, 3,
            D=D, W=W, skips=skips, activation=activation, output_activation=output_activation,
            dtype=dtype, device=device, weight_norm=weight_norm,
        )


        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, v: torch.Tensor, position: torch.Tensor):
        h, w, _ = v.shape
        v_flatten = v.reshape(-1, 3)
        network_input = self.embed_fn_view(v_flatten)
        position_flatten = position.reshape(-1, 3)
        position_embed = self.position_embed(position_flatten)
        positional_embedding = self.position_encoding(position_flatten)
        network_input = torch.cat([network_input, positional_embedding, position_embed], dim=-1)
        return self.blocks(network_input).reshape(h, w, 3)

    def render_with_camera(self, H, W, intr, extr):
        def get_rays(H, W, K, c2w):
            i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                                  torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
            i = i.t().cuda()
            j = j.t().cuda()
            dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].expand(rays_d.shape)
            return rays_o, rays_d

        rays_o, rays_d = get_rays(H, W, intr, extr)

        output_color = self.forward(rays_d, rays_o)

        return output_color.permute(2, 0, 1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))




