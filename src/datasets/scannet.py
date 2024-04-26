from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
import cv2
from numpy.linalg import inv

from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic
)
from src.threedsam.utils.geometry import get_point_cloud
# TODO: test dpt 
from src.dpt.models import DPTDepthModel
from src.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from src.utils.io import read_image
from torchvision.transforms import Compose

class ScanNetDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 dpt_weight_path,
                 dpt_optimize=True,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode
        self.optimize = dpt_optimize

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = dict(np.load(intrinsic_path))

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init DPT
        net_w = 640
        net_h = 480
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.dpt = DPTDepthModel(
            path=dpt_weight_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    
        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
        
        self.dpt.eval()

        if dpt_optimize == True and self.device == torch.device("cuda"):
            self.dpt = self.dpt.to(memory_format=torch.channels_last)
            self.dpt = self.dpt.half()


        self.dpt.to(self.device)

    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.pose_dir,
                       scene_name,
                       'pose', f'{name}.txt')
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        
        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
        img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read the depthmap which is stored as (480, 640)
        if self.mode in ['train', 'val']:
            depth0 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'))
            depth1 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        # predict depth using DPT
        img_color0 = read_image(img_name0)
        img_color1 = read_image(img_name1)
        img_input0 = self.transform({"image": img_color0})["image"]
        img_input1 = self.transform({"image": img_color1})["image"]

        with torch.no_grad():
            sample0 = torch.from_numpy(img_input0).to(self.device).unsqueeze(0)
            sample1 = torch.from_numpy(img_input1).to(self.device).unsqueeze(0)
            if self.optimize and self.device == torch.device("cuda"):
                sample0 = sample0.to(memory_format=torch.channels_last)
                sample1 = sample1.to(memory_format=torch.channels_last)
                sample0 = sample0.half()
                sample1 = sample1.half()

            prediction0 = self.dpt.forward(sample0).squeeze(0)    # (h, w)
            prediction1 = self.dpt.forward(sample1).squeeze(0)
            
            prediction0 *= 1000.0
            prediction1 *= 1000.0 

        # get 3d point cloud
        pts_3d0 = get_point_cloud(prediction0, K_0)    # (h * w, 3)
        pts_3d1 = get_point_cloud(prediction1, K_1)

        data = {
            'image0': image0,   # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'pts_3d0': pts_3d0,    # (h * w, 3)
            'pts_3d1': pts_3d1,  
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'ScanNet',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem_name_0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem_name_1}.jpg'))
        }

        return data
