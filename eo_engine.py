# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import os
import numpy as np
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from torch.autograd import Variable

def encode_onehot(labels, n_classes):
    onehot = torch.FloatTensor(labels.size()[0], n_classes)
    labels = labels.data
    if labels.is_cuda:
        onehot = onehot.cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, kl_criterion=None,
                    adjustments=None,
                    lmbda=0.1,
                    use_contrast=False,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    _lmbda = lmbda

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if len(batch) == 3:
            sar_samples, eo_samples, targets = batch[0], batch[1], batch[2]
                                    
            samples = torch.cat([sar_samples, eo_samples])
            targets = targets.repeat(2)
            eo_input = True
        
        elif len(batch) == 2:
            samples, targets = batch[0], batch[1]
            eo_input = False

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if eo_input:
                outputs, outputs_dist, x_feat, x_dist_feat, confidence = model(samples, eo_input=True)
                            
                # Exp5
                labels_onehot = Variable(encode_onehot(targets, args.nb_classes))
                                
                confidence = F.sigmoid(confidence)
                                
                eps = 1e-12
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                
                b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
                conf = confidence * b + (1 - b)
                
                pred_original = torch.cat([outputs+adjustments, outputs_dist+adjustments])
                                
                pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                                
                ce_loss = criterion(pred_new, targets)
                confidence_loss = torch.mean(-torch.log(confidence))
                ce_loss += _lmbda * confidence_loss
                                
                if 0.3 > confidence_loss.item():
                    _lmbda = _lmbda / 1.01
                elif 0.3 <= confidence_loss.item():
                    _lmbda = _lmbda / 0.99
                
                # Exp2
                # kd_loss = kl_criterion(F.log_softmax(outputs, dim=1), F.softmax(outputs_dist, dim=1))
                
                # Exp3
                kd_loss = kl_criterion(F.log_softmax(outputs, dim=1), F.softmax(outputs_dist.detach(), dim=1))
                
                
                # Exp1
                # x_feat = F.normalize(x_feat, p=2, dim=-1)
                # x_dist_feat = F.normalize(x_dist_feat, p=2, dim=-1)
                # logits_per_x = 1.-torch.matmul(x_feat, x_dist_feat.T)
                # labels = targets[:args.batch_size//2].unsqueeze(0)
                # labels = labels == labels.T
                # labels = labels.float()
                
                # Exp2
                x_feat = F.normalize(x_feat, p=2, dim=-1)
                x_dist_feat = F.normalize(x_dist_feat, p=2, dim=-1)
                
                feat = torch.cat([x_feat, x_dist_feat])
                logits_per_x = 1.-torch.matmul(feat, feat.T)
                labels = targets.unsqueeze(0)
                labels = labels == labels.T
                labels = labels.float()
                
                const_loss = F.binary_cross_entropy_with_logits(logits_per_x, labels)
                
                loss = ce_loss + kd_loss + 10.*const_loss
                
            else:
                if not use_contrast:
                    # Exp4
                    outputs, confidence = model(samples, eo_input=False)
                    
                    labels_onehot = Variable(encode_onehot(targets, args.nb_classes))
                                    
                    confidence = F.sigmoid(confidence)
                                    
                    eps = 1e-12
                    confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                    
                    b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
                    conf = confidence * b + (1 - b)
                    
                    pred_original = outputs+adjustments
                                    
                    pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                                    
                    ce_loss = criterion(pred_new, targets)
                    confidence_loss = torch.mean(-torch.log(confidence))
                    ce_loss += _lmbda * confidence_loss
                    
                    if 0.3 > confidence_loss.item():
                        _lmbda = _lmbda / 1.01
                    elif 0.3 <= confidence_loss.item():
                        _lmbda = _lmbda / 0.99
                        
                    loss = ce_loss
                
                else:
                    # Exp6
                    outputs, x_feat, confidence = model(samples, eo_input=False, use_feat=True)
                    
                    labels_onehot = Variable(encode_onehot(targets, args.nb_classes))
                                    
                    confidence = F.sigmoid(confidence)
                                    
                    eps = 1e-12
                    confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                    
                    b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
                    conf = confidence * b + (1 - b)
                    
                    pred_original = outputs+adjustments
                                    
                    pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                                    
                    ce_loss = criterion(pred_new, targets)
                    confidence_loss = torch.mean(-torch.log(confidence))
                    ce_loss += _lmbda * confidence_loss
                                        
                    if 0.3 > confidence_loss.item():
                        _lmbda = _lmbda / 1.01
                    elif 0.3 <= confidence_loss.item():
                        _lmbda = _lmbda / 0.99
                        
                    x_feat = F.normalize(x_feat, p=2, dim=-1)
                    
                    feat = x_feat
                    logits_per_x = 1.-torch.matmul(feat, feat.T)
                    labels = targets.unsqueeze(0)
                    labels = labels == labels.T
                    labels = labels.float()
                    
                    const_loss = F.binary_cross_entropy_with_logits(logits_per_x, labels)
                        
                    loss = ce_loss + 10.*const_loss
                    
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(ce_loss=ce_loss.item())
        
        if eo_input:
            metric_logger.update(kd_lossx10=kd_loss.item())
            metric_logger.update(const_loss=const_loss.item())
            
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, _lmbda


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]  # TODO: check why default use -1
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Final_Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()), str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())), str(int(split_nb[i].cpu().numpy()))
            )
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, is_hmdb=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video_hmdb if is_hmdb else compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


def compute_video_hmdb(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    # print(feat.shape)
    try:
        pred = np.argmax(feat)
        top1 = (int(pred) == int(label)) * 1.0
        top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    except:
        pred = 0
        top1 = 1.0
        top5 = 1.0
        label = 0
    return [pred, top1, top5, int(label)]