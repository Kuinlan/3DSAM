import torch


def anchor_index_padding(data, mask, 
                         anchor_num_max, 
                         pad_num_min, 
                         is_training):

    N = mask.shape[0]
    device = mask.device 

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    anchor_num = mask.sum(dim=(1, 2)).to(torch.int32)  # [N', ]

    anchor_num_cumsum = torch.cat(
        [torch.tensor([0], dtype=torch.int32, device=anchor_num.device), 
         anchor_num.cumsum(0)], 
        dim=0
    )

    if is_training:
        pad_num = torch.where(
            anchor_num < anchor_num_max - pad_num_min,
            anchor_num_max - pad_num_min - anchor_num,
            0,
        )  # (N', )
        anc_i_gt = data['anchor_i_gt']  # (N', ANCHOR_NUM)
        anc_j_gt = data['anchor_j_gt']
        anc_i_ids = torch.zeros_like(anc_i_gt)  # (N', ANCHOR_NUM)
        anc_j_ids = torch.zeros_like(anc_j_gt)
        anc_i_ids[:, :pad_num_min] = anc_i_gt[:, :pad_num_min]
        anc_j_ids[:, :pad_num_min] = anc_j_gt[:, :pad_num_min]

        for n in range(N):
            low = anchor_num_cumsum[n]            

            anc_i_ids[n, pad_num_min:pad_num_min+pad_num[n]] = anc_i_gt[n, pad_num_min:pad_num_min+pad_num[n]]
            anc_j_ids[n, pad_num_min:pad_num_min+pad_num[n]] = anc_j_gt[n, pad_num_min:pad_num_min+pad_num[n]]

            sample = torch.randperm(anchor_num[n], dtype=torch.int64, device=device) + low

            anc_i_ids[n, pad_num_min+pad_num[n]:] = i_ids[sample[:anchor_num_max-pad_num_min-pad_num[n]]]
            anc_j_ids[n, pad_num_min+pad_num[n]:] = j_ids[sample[:anchor_num_max-pad_num_min-pad_num[n]]]
            
        
        shuffle = torch.randperm(anchor_num_max, dtype=torch.int64, device=device)
        anc_i_gt = anc_i_gt[:, shuffle]
        anc_j_gt = anc_j_gt[:, shuffle]

    else:  
        sample_index = torch.randint(low=0, high=anchor_num[0], size=(1, anchor_num_max), 
                                     dtype=torch.int64, device=device)  
        anc_i_gt = i_ids[sample_index]  # [1, ANCHOR_NUM]
        anc_j_gt = j_ids[sample_index]

    return anc_i_gt, anc_j_gt
    # if is_training:
    #     anchor_ids_gt = data['anchor_ids_gt']  # [N, ANCHOR_NUM]
    #     need_pad = anchor_num < anchor_num_max - pad_num_min

    #     anchor_i_ids = []
    #     anchor_j_ids = []
    #     for n in range(N):
    #         i_ids_pad = data['spv_i_ids'][anchor_ids_gt[n, :pad_num_min]]
    #         j_ids_pad = data['spv_j_ids'][anchor_ids_gt[n, :pad_num_min]]
    #         low = anchor_num_cumsum[n]
    #         high = anchor_num_cumsum[n+1]
    #         if need_pad[n]:
    #             pad_num = anchor_num_max - pad_num_min - anchor_num[n]
    #             i_ids_final = torch.cat(
    #                 [i_ids_pad, 
    #                 data['spv_i_ids'][anchor_ids_gt[n, pad_num_min:pad_num_min+pad_num]],
    #                 i_ids[low:high]])
    #             j_ids_final = torch.cat(
    #                 [j_ids_pad, 
    #                 data['spv_j_ids'][anchor_ids_gt[n, pad_num_min:pad_num_min+pad_num]],
    #                 j_ids[low:high]])
    #         else:
    #             sample_index = (torch.randperm(
    #                 anchor_num[n], dtype=torch.int64, device=device) + low[n])[:anchor_num_max-pad_num_min]
    #             i_ids_final = torch.cat([i_ids_pad, i_ids[sample_index]])
    #             j_ids_final = torch.cat([j_ids_pad, j_ids[sample_index]])

    #         ids_shuffle = torch.randperm(
    #             len(i_ids_final), dtype=torch.int64, device=device
    #         )

            # anchor_i_ids.append(i_ids_final[ids_shuffle])
            # anchor_j_ids.append(j_ids_final[ids_shuffle])

        # anchor_i_ids = torch.stack(anchor_i_ids, dim=0)  # [N, ANCHOR_NUM]
        # anchor_j_ids = torch.stack(anchor_j_ids, dim=0)

