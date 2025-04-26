# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    imgs = args.img
    # test a single image
    import warnings
    from pathlib import Path

    import mmcv
    import numpy as np
    import torch
    from mmcv.ops import RoIPool
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint

    from mmdet.core import get_classes
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    from mmdet.models import build_detector

    import cv2
    import seaborn as sns
    import matplotlib.pyplot as plt

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():

        results = model.get_features(data['img'][0])
        x0, x1, weight = results
        x0, x1, weight = x0.cpu(), x1.cpu(), weight.cpu()

        # show the weight
        weight = weight[0, 0, :, :].numpy()
        x0 = x0[0].transpose(0, 1).sum(1).detach().numpy()
        x1 = x1[0].transpose(0, 1).sum(1).detach().numpy()
        plt.figure(figsize=(10.39, 10.39))
        heatmap = sns.heatmap(x1, cmap='jet', square=True, cbar=False)
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度

        plt.savefig('demo/x1.png', bbox_inches='tight', pad_inches=0)


        plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
