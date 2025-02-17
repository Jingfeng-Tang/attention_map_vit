import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets.cub import CUBDataset
from engine import train_one_epoch, evaluate, generate_attention_maps_ms, generate_bounding_boxes
import utils
import os
import numpy as np
import random
import timm
import models
from utils import metric_format

def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=60, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_tiny_patch16_224.augreg_in21k_ft_in1k', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='VOCdevkit/VOC2012', type=str, help='dataset path')
    parser.add_argument('--img-list', default='voc12', type=str, help='image list path')
    parser.add_argument('--data-set', default='VOC12', type=str, help='dataset')


    parser.add_argument('--output_dir', default='saved_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--ckpt', default='', help='generate bounding boxes from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)


    # generating bounding boxes
    parser.add_argument('--gen_bounding_boxes', default=False, action='store_true')
    parser.add_argument('--gen_attention_maps', default=False, action='store_true')
    parser.add_argument('--gen_maps_boxes', default=False, action='store_true')
    parser.add_argument('--refine_patch_att_mat', default=False, action='store_true')
    parser.add_argument('--attention_maps_dir', type=str, default='attention_maps')
    parser.add_argument('--maps_boxes_dir', type=str, default='maps_boxes')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default='cam-png')
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')

    parser.add_argument('--patch-attn-refine', type=bool, default=True)
    parser.add_argument('--visualize-cls-attn', type=bool, default=True)

    parser.add_argument('--gt-dir', type=str, default=None)
    parser.add_argument('--cam-npy-dir', type=str, default='cam-npy')
    parser.add_argument("--scales", nargs='+', type=float, default=[1.0,0.75,1.25])
    parser.add_argument('--label-file-path', type=str, default=None)
    parser.add_argument('--attention-type', type=str, default='fused')
    parser.add_argument("--att_thr", type=float, default=0.6)


    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument("--loss-weight", default=1.0, type=float)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main(args):
    print(args)
    device = torch.device(args.device)
    # seed = args.seed
    # cudnn.benchmark = True
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # dataset
    dataset_cub_train = CUBDataset('/data/c425/tjf/datasets/CUB_200_2011/', is_train=True, is_gen_bbox = False)
    dataset_cub_val = CUBDataset('/data/c425/tjf/datasets/CUB_200_2011/', is_train=False, is_gen_bbox = False)
    dataset_cub_gen = CUBDataset('/data/c425/tjf/datasets/CUB_200_2011/', is_train=False, is_gen_bbox = True)

    sampler_train = torch.utils.data.RandomSampler(dataset_cub_train)

    # dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_cub_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_cub_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    data_loader_gen = torch.utils.data.DataLoader(
        dataset_cub_gen,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    print(f"Creating pretrained model: {args.model} ")

    # model = create_model(
    #     args.model,
    #     pretrained=False,
    #     num_classes=200,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    #     checkpoint_path=args.finetune
    # )

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=200,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        checkpoint_path=args.finetune
    )

    # model_p = timm.create_model(
    #     'vit_small_patch16_224.augreg_in21k_ft_in1k',
    #     pretrained=False,
    #     num_classes=200,
    #     drop_rate=args.drop,
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=None,
    #     checkpoint_path=args.finetune
    # )


    # print(model)

    # print('----------------------------------------------')
    # print(model_p)
    # for i, block in enumerate(model.blocks.children()):
    #     print(f'block: {block.attn.proj.weight}')
    #
    # state_dict = model.state_dict()
    # state_dict_p = model_p.state_dict()

    # model.load_state_dict(state_dict_p, strict=True)

    # print(type(state_dict))
    # print(state_dict == state_dict_p) 
    # print('-------888888888888888888888888888888888888--------------------------------------------------------------------------------------------')
    # print(state_dict_p)
    # for k, v in state_dict.items():
    #     if k in state_dict_p:
    #         # print(f'k: {k}\nstate_dict[k]: {state_dict[k]}')
    #         if torch.equal(state_dict[k], state_dict_p[k]):
    #             # print('ok')
    #             pass
    #         else:
    #             print(f'k: {k}')
    #             # model.state_dict()[k] = state_dict_p[k]
    #             state_dict[k] = state_dict_p[k]
    #             print('------------diff----------')
    #     else:
    #         print('no k')
    #
    # model.load_state_dict(state_dict, strict=True)
    #
    # new_state_dict = model.state_dict()
    # print('------------dif99999999999999999999999999999999f----------')
    #
    # for k, v in new_state_dict.items():
    #     if k in state_dict_p:
    #         # print(f'k: {k}\nstate_dict[k]: {state_dict[k]}')
    #         if torch.equal(state_dict[k], state_dict_p[k]):
    #             # print('ok')
    #             pass
    #         else:
    #             # print(f'k: {k}')
    #             # # model.state_dict()[k] = state_dict_p[k]
    #             # state_dict[k] = state_dict_p[k]
    #             print('------------diff----------')
    #     else:
    #         print('no k')


    # for k, v in state_dict.items():
    #     if state_dict[k] != state_dict_p[k]:
    #         print('------------diff----------')


    # a = []
    # b = a[1]


    # state_dict = model.state_dict()
    # for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]
    # print(model.default_cfg)  # 查看模型cfg

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
    #     return

    if args.gen_bounding_boxes:
        output_dir = Path(args.output_dir)
        print(f"------------------------Start gen_bounding_boxes------------------------\n")
        with (output_dir / "log.txt").open("a") as f:
            f.write(f"------------------------Start gen_bounding_boxes------------------------\n")

        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        cls_top1, cls_top5, mIoU_top1, correct_samples_mIoU_top1 = generate_bounding_boxes(data_loader_gen, model, device, args)
        res = {
            "cls_top1": cls_top1,
            "cls_top5": cls_top5,
            "mIoU_top1": mIoU_top1 * 100,
            "correct_samples_mIoU_top1": correct_samples_mIoU_top1 * 100,
        }
        table = metric_format(res)
        print(table.draw())
        print(f"\n------------------------End gen_bounding_boxes------------------------\n")
        with (output_dir / "log.txt").open("a") as f:
            f.write(table.draw())
            f.write(f"\n------------------------End gen_bounding_boxes------------------------")

        with (Path('./auto_run_res.txt')).open("a") as f:
            f.write(f"\n++++++++++++++++++++++++Start auto_run_results++++++++++++++++++++++++\n")
            f.write(f'args.att.thr: {args.att_thr}\n')
            f.write('\n')
            f.write(table.draw())
            f.write(f"\n------------------------End auto_run_results------------------------\n")
        # with open("auto_run_metric_log.txt", "w") as f:
        #     f.write(f"------------------------Start gen_bounding_boxes------------------------\n")
        #     f.write(f"att_thr: {args.att_thr:.3f}\n")
        #     f.write(f"cls_top1: {cls_top1:.3f}\n")
        #     f.write(f"cls_top5: {cls_top5:.3f}\n")
        #     f.write(f"mIoU_top1: {mIoU_top1 * 100:.3f}\n")
        #     f.write(f"correct_samples_mIoU_top1: {correct_samples_mIoU_top1 * 100:.3f}\n")
        #     f.write(f"------------------------End gen_bounding_boxes------------------------")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_top1=0.0
    best_top5=0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            args=args
        )

        lr_scheduler.step(epoch)

        # test_stats = evaluate(data_loader_val, model, device)
        top1, top5 = evaluate(data_loader_val, model, device)
        print('+++++++++++++++++++++++++++++++++++++++')
        print(f'top1: {top1}\ntop5: {top5}')
        print('+++++++++++++++++++++++++++++++++++++++')
        if top1 > best_top1:
            best_top1 = top1
        if top5 > best_top5:
            best_top5 = top5

        if args.output_dir and top1 >= best_top1:
            checkpoint_paths = [output_dir / 'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'top1': str(top1),
                     'top5': str(top5),
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        # ckpt_name = 'checkpoint_epoch_' + str(epoch) + '.pth'
        # print(f'ckpt_name: {ckpt_name}')
        # torch.save({'model': model.state_dict()}, output_dir / ckpt_name)

    # torch.save({'model': model.state_dict()}, output_dir / 'checkpoint.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('++++++++++++++---------Best----------+++++++++++++++++++++')
    print(f'best_top1: {best_top1}\nbest_top5: {best_top5}')
    print('++++++++++++++---------Best----------+++++++++++++++++++++')


if __name__ == '__main__':
    same_seeds(0)

    parser = argparse.ArgumentParser('ViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    start_time = datetime.datetime.now()
    formatted_date_time = start_time.strftime("%Y-%m-%d-%H-%M-%S")

    if args.gen_bounding_boxes:
        formatted_date_time = formatted_date_time + '_gen_bbox'

    if args.gen_maps_boxes:
        formatted_date_time = formatted_date_time + '_gen_maps_boxes'

    if args.refine_patch_att_mat:
        formatted_date_time = formatted_date_time + '_refine'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        args.output_dir = os.path.join(args.output_dir, formatted_date_time)
        os.makedirs(args.output_dir)

    if args.gen_attention_maps:
        args.attention_maps_dir = os.path.join(args.output_dir, args.attention_maps_dir)
        os.makedirs(args.attention_maps_dir)

    if args.gen_maps_boxes:
        args.maps_boxes_dir = os.path.join(args.output_dir, args.maps_boxes_dir)
        os.makedirs(args.maps_boxes_dir)


    output_dir = Path(args.output_dir)
    with (output_dir / "log.txt").open("a") as f:
        f.write(f"------------------------Start Parameters------------------------\n")
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
        f.write(f"------------------------End Parameters------------------------\n")

    main(args)
