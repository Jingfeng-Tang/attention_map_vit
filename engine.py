import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 255, 255, 255, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(f'samples: {samples[0,0,0,0]}')  # [128]

        # print(f'targets.shape: {targets.shape}')  # [128]
        # print(f'targets.shape: {targets}')  # [128]

        patch_outputs = None
        c_outputs = None
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            # print(f'outputs: {outputs[0, 0]}')  # [128]
            # a = []
            # b = a[1]
            # print(f'sample.shape: {samples.shape}') # [128, 3, 224, 224]
            # print(f'outputs.shape: {outputs.shape}')    # [128, 200]
            cls_loss = cls_criterion(outputs, targets)
            metric_logger.update(cls_loss=cls_loss.item())

        loss_value = cls_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(cls_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    cls_top1 = []
    cls_top5 = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for ori_img, images, target, gt_bbox, name in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # print(f'images.shape: {images.shape}')  # [128, 3, 224, 224]
        # print(f'target.shape: {target.shape}')  # [128]
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            # print(f'output.shape: {output.shape}')  # [128, 200]
            loss = criterion(output, target)
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
            cls_top1.append(prec1.cpu().numpy())
            cls_top5.append(prec5.cpu().numpy())
        # metric_logger.update(loss=loss.item())
        # print('ok5')

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('ok6')

    # print(f'top1 {np.mean(cls_top1)} top5 {np.mean(cls_top5)}')

    return np.mean(cls_top1), np.mean(cls_top5)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    img_list = open(os.path.join(args.img_list, 'train_aug_id.txt')).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                else:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index,
                                                               attention_type=args.attention_type)
                    patch_attn = torch.sum(patch_attn, dim=0)


                if args.patch_attn_refine:
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)

                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    # norm_cam = np.zeros((args.nb_classes, w_orig, h_orig))
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cls_attention = sum_cam[b,cls_ind,:]

                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention
                            # norm_cam[cls_ind] = cls_attention

                            # if args.attention_dir is not None:
                            #     fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                            #     show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return

@torch.no_grad()
def generate_bounding_boxes(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating bounding boxes:'

    model.eval()

    loc_gt_known = []
    loc_top1 = []
    cls_top1 = []
    cls_top5 = []

    for ori_img, images, target, gt_bbox, name in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # gt_bbox = gt_bbox.to(device, non_blocking=True)

        # print(gt_bbox)
        b, h, w, d = ori_img.shape
        # print(f'ori_img: {name}')
        # a = []
        # b = a[1]

        with torch.cuda.amp.autocast():
            x_cls_logits, att_map = model(images, return_att=True, n_layers=12)

            prec1, prec5 = utils.accuracy(x_cls_logits.data, target, topk=(1, 5))
            cls_top1.append(prec1.cpu().numpy())
            cls_top5.append(prec5.cpu().numpy())
            # cls correct
            value, indice = x_cls_logits.data.topk(1, 1, True, True)
            indices = indice.t()
            correct = indices.eq(target.view(1, -1).expand_as(indices))

            # print(f'x_cls_logits: {x_cls_logits.shape}')    # [1, 200]
            # print(f'att_map: {att_map.shape}')  # [1, 14, 14]
            att_map = att_map.unsqueeze(0)
            # resize
            cls_attention = F.interpolate(att_map, size=(h, w), mode='bilinear', align_corners=False)[0,0,:,:]
            # print(f'cls_attentions: {cls_attentions.shape}')  # [224,224]
            # normalize
            # cls_attentions = F.relu(cls_attentions)
            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)

            # a = []
            # b = a[1]

            # loc_top1
            if correct:
                # Estimate BBOX
                gt_bbox = gt_bbox[0].strip().split(' ')
                gt_bbox = list(map(float, gt_bbox))
                cls_attention = cls_attention.cpu().numpy()
                estimated_bbox = get_bboxes(cls_attention, args.att_thr)
                iou = cal_iou(estimated_bbox, gt_bbox)
                loc_top1.append(iou)
            else:
                loc_top1.append(0.0)
            loc_top1_miou = np.mean(loc_top1)





            if args.gen_attention_maps:
                name_str = name[0]
                name_str = name_str.replace(".jpg", "")
                name_str = name_str.replace("/", "_")
                fname = os.path.join(args.attention_maps_dir, name_str + '_class_' + str(indice[0].item()) + '_score_' + str(value[0].item()) + '_' + str(correct[0].item()) + '.png')

                ori_img = ori_img.squeeze(0).cpu().numpy()
                cls_attention = cls_attention.cpu().numpy()

                show_cam_on_image(ori_img, cls_attention, fname)





            # iou_i = cal_iou(estimated_bbox, gt_bbox)

            # # gt known
            # if iou_i >= 0.5:
            #     loc_gt_known.append(1)
            # else:
            #     loc_gt_known.append(0)

    # gt_known = list2acc(loc_gt_known)

    # top1, top5 = np.mean(cls_top1), np.mean(cls_top5)

    # a = []
    # b = a[1]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return loc_top1_miou


def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox


def cal_iou(box1, box2, method='iou'):
    """
    support:
    1. box1 and box2 are the same shape: [N, 4]
    2.
    :param box1:
    :param box2:
    :return:
    """

    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]

    iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0]) + 1
    ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1]) + 1

    i_area = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    if method == 'iog':
        iou_val = i_area / (box2_area)
    elif method == 'iob':
        iou_val = i_area / (box1_area)
    else:
        iou_val = i_area / (box1_area + box2_area - i_area)
    return iou_val


def list2acc(results_list):
    """
    :param results_list: list contains 0 and 1
    :return: accuarcy
    """
    accuarcy = results_list.count(1)/len(results_list)
    return accuarcy



def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)