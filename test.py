#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from uni3r.utils.path_manager import init_all_submodules
init_all_submodules()
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# Model
from uni3r.uni3r import Uni3R
# Dataset
from uni3r.datasets.testdata import TestDataset  # noqa
import dust3r.datasets
dust3r.datasets.TestDataset = TestDataset
# Loss
from uni3r.loss import loss_of_one_batch  # noqa
from uni3r.loss import TestLoss
import dust3r.losses
dust3r.losses.TestLoss = TestLoss

from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from uni3r.loss import loss_of_one_batch  # noqa

import croco.utils.misc as misc  # noqa

def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R testing', add_help=False)
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--lseg_pretrained', default="checkpoints/pretrained_models/demo_e200.ckpt", help='path of lseg_pretrained')
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")
    
    # testing parameters
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size per GPU")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1], help="Use Automatic Mixed Precision")
    parser.add_argument('--print_freq', default=20, type=int, help='frequency to print infos while testing')
    
    # output directories
    parser.add_argument('--test_results_dir', default='./test_results/', type=str, help="path where to save the test results")
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print("test_results_dir: "+args.test_results_dir)
    if args.test_results_dir:
        Path(args.test_results_dir).mkdir(parents=True, exist_ok=True)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build test dataset
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # Load model and criterion
    print(f'>> Creating test criterion = {args.test_criterion}')
    test_criterion = eval(args.test_criterion).to(device)

    import re
    dataset_string = args.test_dataset
    pattern = r"num_views=(\d+)"
    match = re.search(pattern, dataset_string)
    if match:
        num_views_str = match.group(1)
        num_views_value = int(num_views_str)
    else:
        raise ValueError("num_views not found in test_dataset")
    
    model = Uni3R.from_pretrained(args.pretrained, device, num_views_value=num_views_value)
    model.eval()
    
    # Test on datasets
    test_stats = {}
    for test_name, testset in data_loader_test.items():
        stats = test_one_epoch(model, test_criterion, testset,
                             device, 1, args=args, prefix=test_name)
        test_stats[test_name] = stats

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # fix the epoch for the dataset
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(1) # 1 is a dummy value
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(1) # 1 is a dummy value

    total_time = []
    for batch_id, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        res = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=False,
                                       use_amp=bool(args.amp),
                                       total_time=total_time)
        loss_tuple = res['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values

        # results = loss_details.pop('results')
        metric_logger.update(loss=float(loss_value), **loss_details)

    print('PSNR: ', criterion.psnr.compute())
    print('SSIM: ', criterion.ssim.compute())
    print('LPIPS: ', criterion.compute_lpips_mean())
    print('mIoU: ', criterion.miou.compute(), criterion.miou.compute().mean())
    print('Acc: ', criterion.accuracy.compute())
    print('Depth Metrics: ', criterion.depth_metric.compute())

    sum_time = 0.0
    avg_time = 0.0
    if len(total_time) > 0:
        for i in range(len(total_time)):
            sum_time += total_time[i]
        avg_time = sum_time / len(total_time)
    print(f"\n⚡ Average Inference Time: {avg_time:.4f} seconds per scene")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)
        
    return results

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
