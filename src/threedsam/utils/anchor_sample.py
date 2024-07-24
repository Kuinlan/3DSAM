import torch
from src.threedsam.utils.geometry import estimate_pose, get_scaled_K

# def anchor_padding_topK(mask, anchor_num_max, data, is_training=False):
#     N = mask.shape[0]
#     device = mask.device 

#     mask_v, all_j_ids = mask.max(dim=2)
#     b_ids, i_ids = torch.where(mask_v)
#     j_ids = all_j_ids[b_ids, i_ids]
#     anchor_num = mask.sum(dim=(1, 2)).to(torch.int32)  # [N', ]

#     anc_i_gt = data['anchor_i_gt']
#     anc_j_gt = data['anchor_j_gt']
#     conf_matrix = data['conf_matrix']
#     mconf = conf_matrix[b_ids, i_ids, j_ids]
    
#     i_ids, j_ids = torch.split(i_ids, anchor_num.tolist()), torch.split(j_ids, anchor_num.tolist())
#     mconf = torch.split(mconf, anchor_num.tolist())

#     # pick top k matches
#     for n in range(N):
#         if anchor_num[n] >= anchor_num_max:
#             i_ids_split, j_ids_split = i_ids[n], j_ids[n]
#             mconf_split = mconf[n]
#             _, indices = torch.sort(mconf_split, descending=True)
#             indices = indices[:anchor_num_max]
#             i_ids_split, j_ids_split = i_ids_split[indices], j_ids_split[indices]
#         else:
#             if is_training:
#                 pad_num = anchor_num_max - anchor_num[n]
#                 i_ids_split = torch.cat([i_ids[n], anc_i_gt[n][:pad_num]])
#                 j_ids_split = torch.cat([j_ids[n], anc_j_gt[n][:pad_num]])

#                 sample = torch.randperm(anchor_num_max, dtype=torch.int64, device=i_ids_split.device)
#                 i_ids_split = i_ids_split[sample]
#                 j_ids_split = j_ids_split[sample]
#             else:
#                 sample = torch.randint(low=0, high=anchor_num[n], size=(anchor_num_max, ), dtype=torch.int64, device=device)
#                 i_ids_split = i_ids[n][sample]
#                 j_ids_split = j_ids[n][sample]

#         i_ids[n], j_ids[n] = i_ids_split, j_ids_split

#     anc_i_ids, anc_j_ids = torch.stack(i_ids, dim=0), torch.stack(j_ids, dim=0)  # [N, ANCHOR_NUM]

#     return anc_i_ids, anc_j_ids

def get_anchor(b_ids, i_ids, j_ids, mconf, 
                anchor_num_max, is_training, data):

    non_skip_ids = data['non_skip_ids']
    batch_size = non_skip_ids.shape[0]
    scale = data['hw0_i'][0] // data['hw0_c'][0]

    K0 = get_scaled_K(data['K0'][non_skip_ids].clone(), scale)
    K1 = get_scaled_K(data['K1'][non_skip_ids].clone(), scale)

    # get 2D coordinate for pose estimation
    pts0 = torch.stack([i_ids % data['hw0_c'][1], 
                        i_ids // data['hw0_c'][1]], dim=-1).to(torch.float32)  
    pts1 = torch.stack([j_ids % data['hw1_c'][1], 
                        j_ids // data['hw1_c'][1]], dim=-1).to(torch.float32)
    device = pts0.device

    anc_i_ids = torch.zeros((batch_size, anchor_num_max), device=device, dtype=torch.int64)
    anc_j_ids = torch.zeros((batch_size, anchor_num_max), device=device, dtype=torch.int64)
    R = torch.zeros((batch_size, 3, 3), device=device)
    t = torch.zeros((batch_size, 3, 1), device=device)

    if is_training:
        anc_i_gt = data['anchor_i_gt'][non_skip_ids]  # (N, ANCHOR_NUM)
        anc_j_gt = data['anchor_j_gt'][non_skip_ids]

        R_gt = data['T_0to1'][non_skip_ids, 0:3, 0:3]  # [N, 3, 3]
        t_gt = data['T_0to1'][non_skip_ids, 0:3, 3].unsqueeze(dim=-1)    # [N, 3, 1]

        for n in range(batch_size):
            mask = b_ids == n
            i_ids_n = i_ids[mask]
            j_ids_n = j_ids[mask]
            pts0_n = pts0[mask]
            pts1_n = pts1[mask]
            weight_n = mconf[mask]
            match_num_n = pts0_n.shape[0]

            if match_num_n >= anchor_num_max:
               sample = torch.randperm(match_num_n, device=device, dtype=torch.int64)[:anchor_num_max]
               anc_i_ids[n] = i_ids_n[sample]
               anc_j_ids[n] = j_ids_n[sample]
            else:
                if match_num_n > 0:
                    sample = torch.randint(low=0, high=match_num_n, size=(anchor_num_max, ), device=device)
                    anc_i_ids[n] = i_ids_n[sample]
                    anc_j_ids[n] = j_ids_n[sample]
                else:
                    anc_i_ids[n] = anc_i_gt[n]
                    anc_j_ids[n] = anc_j_gt[n]

            # pose estimation
            if match_num_n >= 8:
                R[[n]], t[[n]] = estimate_pose(pts0_n[None], pts1_n[None], K0[n][None], K1[n][None], weight_n[None])  # [1, 3, 3], [1, 3, 1] 
            else:
                R[n], t[n] = R_gt[n], t_gt[n]

    # eval/test
    else:  
        match_num = pts0.shape[0]
        if match_num >= anchor_num_max:
            sample = torch.randperm(match_num, dtype=torch.int64, device=device)[:anchor_num_max]  
        else:
            sample = torch.randint(low=0, high=match_num, size=(anchor_num_max, ), device=device)

        anc_i_ids[0] = i_ids[sample] 
        anc_j_ids[0] = j_ids[sample]

        if (match_num >= 8):
            R, t = estimate_pose(pts0[None], pts1[None], K0, K1, mconf[None])
        else:
            R = torch.eye(3, device=device)[None] 
            t = torch.zeros(size=(3, 1), device=device)[None]

    return anc_i_ids, anc_j_ids, R, t