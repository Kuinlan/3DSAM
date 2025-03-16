import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors
import cv2 
import torch
import os
from einops.einops import rearrange

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None, batch_idx=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')

    # axes[0].imshow(img0[:,:,[2,1,0]])
    # axes[1].imshow(img1[:,:,[2,1,0]])
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path)+"/"+str(batch_idx))
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', path=None, batch_idx=None, require_depth=False):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'MDFEM',
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path=path, batch_idx=batch_idx)

    if require_depth:
        save_depth_visualization(data, b_id, '/public/home/szlsygb/Hu/3DSAM/output/MDFEM_3DPPE_DA_est_pose/depthmap/'+str(batch_idx)+'.png')
    return figure

def make_pos_embedding_similarity_map(data, path, batch_idx):
    # 设置随机数种子
    torch.manual_seed(88)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(88)
    np.random.seed(88)
    embedding = data['pos_embed0']  # 假设 shape 为 (1, H, W, C)
    _, H, W, C = embedding.shape  # [1, 60, 80, 256]

    # 随机选择一个特征向量
    random_h = torch.randint(0, H, (1,))
    random_w = torch.randint(0, W, (1,))
    selected_feature = embedding[0, random_h, random_w, :]  # [1, C]

    # 将特征图展平为 [H*W, C]
    feature_map_flat = embedding.view(-1, C)

    # 计算余弦相似性
    similarity = torch.cosine_similarity(selected_feature, feature_map_flat)  # [H*W]
    similarity = similarity.view(H, W).detach().cpu().numpy()

    # 归一化相似性值到 [0, 1]，以便可视化
    similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())

    # **将原图 resize 到 1/8 以匹配特征图尺寸**
    image = data['image0'].squeeze().detach().cpu().numpy() * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 转换为 RGB 格式
    image_resized = cv2.resize(image, (W, H))  # 调整尺寸至 1/8 大小

    # **相似性热图，使用 COLORMAP_HOT 让高相似度区域更亮**
    similarity_colormap = cv2.applyColorMap((similarity * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # **透明度混合**
    overlay = cv2.addWeighted(image_resized, 0.5, similarity_colormap, 0.5, 0)

    # **绘制结果**
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(overlay)

    # **绘制颜色条**
    norm = mcolors.Normalize(vmin=0, vmax=1)  # 归一化 0-1 范围
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # 仅用于 colorbar，不用于绘图
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Similarity Score")

    # **绘制选定的特征点**
    ax.scatter(random_w, random_h, c='cyan', s=50, marker='*', label="Selected Feature Point")
    ax.legend()

    ax.set_title(f'Feature Similarity Map (Batch {batch_idx})')

    # **调整布局**
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # **保存图片**
    output_path = os.path.join(path, f"{batch_idx}.png")
    plt.savefig(output_path)
    plt.close()

def _make_confidence_figure(data, b_id, path=None, batch_idx=None):
    b_mask = data['m_bids'] == b_id
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()
    color = cm.jet(mconf)
    text = [
        'MDFEM',
        'Matches: {}'.format(len(kpts0)),
    ]
    fig = make_matching_figure(img0, img1, kpts0, kpts1, color, text=text, path=path, batch_idx=batch_idx)

    return fig

def make_matching_figures(data, config, mode='evaluation', path=None, batch_idx=None, require_depth=False):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA,
                path=path, batch_idx=batch_idx, require_depth=require_depth)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id, path=path, batch_idx=batch_idx)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures

def save_depth_visualization(data, b_id, save_path):
    """
    可视化并保存深度图，将 depthmap0 与 depthmap1 左右拼接在一起，并保存到指定路径。
    
    参数：
        data: 数据字典，包含 'depthmap0' 和 'depthmap1'
        b_id: 批次 ID，选择对应图像的索引
        save_path: 保存拼接深度图的路径
    """

    # 读取深度图并转换为 numpy 数组
    depthmap0 = data['depth_map0'][b_id].cpu().numpy()
    depthmap1 = data['depth_map1'][b_id].cpu().numpy()

    # 归一化到 [0, 255] 以进行可视化
    depthmap0 = (depthmap0 - depthmap0.min()) / (depthmap0.max() - depthmap0.min()) * 255
    depthmap1 = (depthmap1 - depthmap1.min()) / (depthmap1.max() - depthmap1.min()) * 255

    depthmap0 = depthmap0.astype(np.uint8)
    depthmap1 = depthmap1.astype(np.uint8)

    # 伪彩色映射（使用 JET 颜色映射）
    depthmap0_color = cv2.applyColorMap(depthmap0, cv2.COLORMAP_OCEAN)
    depthmap1_color = cv2.applyColorMap(depthmap1, cv2.COLORMAP_OCEAN)

    # 左右拼接两个深度图
    depthmap_combined = cv2.hconcat([depthmap0_color, depthmap1_color])

    # 保存到指定路径
    cv2.imwrite(save_path, depthmap_combined)

def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

