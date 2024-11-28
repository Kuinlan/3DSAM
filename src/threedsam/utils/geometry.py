import torch
import cv2
import numpy as np
from kornia.utils import create_meshgrid
from kornia.geometry.epipolar import find_fundamental, decompose_essential_matrix


# def estimate_pose_cv(kpts0: torch.Tensor, kpts1: torch.Tensor, K: torch.Tensor):
#     kpts0 = kpts0.cpu().numpy()
#     kpts1 = kpts1.cpu().numpy()
#     K = K.cpu().numpy()
#     E, mask = cv2.findEssentialMat(kpts0, kpts1, K)
#     _, R, t, mask = cv2.recoverPose(kpts0, kpts1, )
#     R = torch.from_numpy(R)
#     t = torch.from_numpy(t)

#     return R

def estimate_pose_np(kpts0, kpts1, K0, K1, thresh=0.5, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

@torch.no_grad()
def get_scaled_K(K: torch.Tensor, scale):
    print(K)
    K = K.clone()
    if K.dim() == 2:
        K[:2, :] = K[:2, :] / scale
    elif K.dim() == 3:
        K[:, :2, :] = K[:, :2, :] / scale
    else:
        raise ValueError("Expected tensor of shape: [N, 3, 3] or [3, 3]")

    return K

@torch.no_grad()
def estimate_pose(kpts0: torch.Tensor, kpts1: torch.Tensor, K0, K1, weights=None):
    F = find_fundamental(kpts0, kpts1, weights=weights)  # (N, 3, 3) 
    E = K1.transpose(1, 2) @ F @ K0  
    R, _, t = decompose_essential_matrix(E)
    return R, t

@torch.no_grad()
def get_point_cloud(depth, K, scale = 8):
    """
    Args:
        depth (torch.Tensor): [N, h, w]
        K (torch.Tensor): [N, 3, 3]
        scale (int)

    Returns:
        pts_3d: (torch.Tensor): [N, L, 3]
    """

    _device = depth.device
    N = depth.shape[0]
    h0 = depth.shape[1] 
    w0 = depth.shape[2] 
    grid_pt = create_meshgrid(h0, w0, False, device = _device).reshape(1, h0*w0, 2).repeat(N, 1, 1)  # (N, h * w, 2)
    grid_pt *= scale

    depth = depth.reshape(N, h0 * w0) # [N, h * w]
     
    # Unproject
    grid_pt_h = torch.cat([grid_pt, torch.ones_like(grid_pt[:, :, [0]])], dim=-1) * depth[..., None]  # (N, h * w, 3)
    
    # (K.inv() @ P.T).T = P @ k.inv().T
    pts_3d = grid_pt_h @ K.inverse().transpose(1, 2)  # (N, L, 3)
    
    return pts_3d


@torch.no_grad()
def get_epipolar_line_kb(coord_2d, R, T, K_ref, K_src):
    """get epipolar lines correspondent to 2D points
    Args:
        coord_2d (torch.Tensor): [N, L, 2] - <x, y>
        R (torch.Tensor): [N, 3, 3]
        t (torch.Tensor): [N, 3, 1]
        K (torch.Tensor): [N, 3, 3]
    Returns:
        lines (torch.Tensor): [N, L, 2] - <k, b>
    Notes:
        K needs to be scaled if coord not in original resolution
    """
    # 齐次形式和反投影
    ones = torch.ones(coord_2d.size(0), coord_2d.size(1), 1, device=coord_2d.device)
    points_3d = torch.cat((coord_2d, ones), dim=2)  # [N, L, 3]
    points_3d = torch.bmm(K_ref.inverse(), points_3d.unsqueeze(-1))  # [N, L, 3, 1]

    # 转换坐标系
    P_hat = torch.bmm(R, points_3d) + T  # [N, L, 3, 1]

    # 一般的二维点投影至源图像
    P_hat_image = torch.bmm(K_src, P_hat).squeeze(-1)  # [N, L, 3]
    c_P_hat = P_hat_image[:, :, :2] / P_hat_image[:, :, 2:]  # [N, L, 2]

    # 中心点投影
    p_M1 = torch.bmm(K_src, T).squeeze(-1)  # [N, 3]
    c_M1 = p_M1[:, :2] / p_M1[:, 2:]  # [N, 2]

    k = (c_P_hat[:, :, 1] - c_M1[:, 1].unsqueeze(1)) / (c_P_hat[:, :, 0] - c_M1[:, 0].unsqueeze(1))
    b = c_M1[:, 1] - (k * c_M1[:, 0])

    return torch.stack((k, b), dim=2)  # [N, L, 2]

@torch.no_grad()
def get_epipolar_line_std(coord_2d, R, t, K_ref, K_src):
    """get epipolar lines correspondent to 2D points
    Args:
        coord_2d (torch.Tensor): [N, L, 2] - <x, y>
        R (torch.Tensor): [N, 3, 3]
        t (torch.Tensor): [N, 3, 1]
        K_ref (torch.Tensor): [N, 3, 3]
        K_sec (torch.Tensor): [N, 3, 3]
    Returns:
        lines (torch.Tensor): [N, L, 3] - <A, B, C>
        mode (torch.Tensor): [N, L]
    Notes:
        K needs to be scaled if coord is not in original scale
    """
    # unproject 
    N, L, _ = coord_2d.shape
    ones = torch.ones((N, L, 1), device=coord_2d.device)
    points_3d = torch.cat((coord_2d, ones), dim=2)  # [N, L, 3]
    points_3d = K_ref.inverse() @ points_3d.transpose(-1, -2)  # [N, 3, L]

    # rigid transform
    P_hat = R @ points_3d + t   # [N, 3, L]

    # project
    P_hat_image = (K_src @ P_hat).transpose(-1, -2)  # [N, L, 3]
    c_P_hat = P_hat_image[:, :, :2] / (P_hat_image[:, :, [2]] + 1e-4)  # [N, L, 2]

    # centor point project
    p_M1 = (K_src @ t).squeeze(-1)  # [N, 3] 
    c_M1 = p_M1[:, :2] / (p_M1[:, [2]] + 1e-4)  # [N, 2]

    # Ax + By + C = 0
    A = c_P_hat[:, :, 1] - c_M1[:, 1].unsqueeze(1)  # [N, L]
    B = c_M1[:, 0].unsqueeze(1) - c_P_hat[:, :, 0]
    C = (c_M1[:, 1].unsqueeze(1) * c_P_hat[:, :, 0] - 
         c_P_hat[:, :, 1] * c_M1[:, 0].unsqueeze(1))

    # y mode: |k| < 1 
    x_dist = torch.abs(B)  # [N, L]
    y_dist = torch.abs(A) 
    y_mode = x_dist > y_dist

    lines = torch.stack([A, B, C], dim=-1)
    lines = lines / lines.norm(p=2, dim=-1, keepdim=True)
        
    return lines, y_mode  # [N, L, 3] - <A, B, C>
                          # [N, L] - <True/False>

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0