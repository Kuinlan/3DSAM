"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import utils.io

from torchvision.transforms import Compose

from src.dpt.models import DPTDepthModel
from src.dpt.transforms import Resize, NormalizeImage, PrepareForNet

#from util.misc import visualize_attention


def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: %s" % device)

    # load network
    net_w = 640
    net_h = 480

    model = DPTDepthModel(
        path=model_path,
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    # normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # transform = Compose(
    #     [
    #         Resize(
    #             net_w,
    #             net_h,
    #             resize_target=None,
    #             keep_aspect_ratio=True,
    #             ensure_multiple_of=32,
    #             resize_method="minimal",
    #             image_interpolation_method=cv2.INTER_CUBIC,
    #         ),
    #         normalization,
    #         PrepareForNet(),
    #     ]
    # )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.io.read_image(img_name)

        if args.kitti_crop is True:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        img_input = transform({"image": img})["image"]

        # computeself.dpt = DPTDepthModel(
            path=dpt_weight_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if model_type == "dpt_hybrid_kitti":
                prediction *= 256

            if model_type == "dpt_hybrid_nyu":
                prediction *= 1000.0

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
