import torch


def anchor_index_padding(data, anchor_num, index, anchor_num_set, 
                         pad_num_min, is_training):
    N = anchor_num.shape[0]
    device = anchor_num.device 
    i_ids_get, j_ids_get = index

    anchor_num_cumsum = torch.cat(
        [torch.tensor([0], dtype=torch.int32, device=anchor_num.device), 
         anchor_num.cumsum(0)], 
        dim=0
    )

    # info prepared in advance to be used to pad
    if is_training:
        anchor_ids_gt = data['anchor_ids_gt']  # [N, ANCHOR_NUM]
        need_pad = anchor_num < anchor_num_set - pad_num_min

        anchor_i_ids = []
        anchor_j_ids = []
        for n in range(N):
            i_ids = data['spv_i_ids'][anchor_ids_gt[n, :pad_num_min]]
            j_ids = data['spv_j_ids'][anchor_ids_gt[n, :pad_num_min]]
            low = anchor_num_cumsum[n]
            high = anchor_num_cumsum[n+1]
            if need_pad[n]:
                pad_num = anchor_num_set - pad_num_min - anchor_num[n]
                i_ids = torch.cat(
                    [i_ids, 
                    data['spv_i_ids'][anchor_ids_gt[n, pad_num_min:pad_num_min+pad_num]],
                    i_ids_get[low:high]])
                j_ids = torch.cat(
                    [j_ids, 
                    data['spv_j_ids'][anchor_ids_gt[n, pad_num_min:pad_num_min+pad_num]],
                    j_ids_get[low:high]])
            else:
                sample_index = (torch.randperm(
                    anchor_num[n], dtype=torch.int64, device=device) + low[n])[:anchor_num_set-pad_num_min]
                i_ids = torch.cat([i_ids, i_ids_get[sample_index]])
                j_ids = torch.cat([j_ids, j_ids_get[sample_index]])

            ids_shuffle = torch.randperm(
                len(i_ids), dtype=torch.int64, device=device
            )

            anchor_i_ids.append(i_ids[ids_shuffle])
            anchor_j_ids.append(j_ids[ids_shuffle])

        anchor_i_ids = torch.stack(anchor_i_ids, dim=0)  # [N, ANCHOR_NUM]
        anchor_j_ids = torch.stack(anchor_j_ids, dim=0)

    else:  # eval/test
        sample_index = torch.randint(low=0, high=anchor_num[0], size=(1, anchor_num_set), 
                                     dtype=torch.int64, device=device)  
        anchor_i_ids = i_ids[sample_index]  # [1, ANCHOR_NUM]
        anchor_j_ids = j_ids[sample_index]

    return anchor_i_ids, anchor_j_ids