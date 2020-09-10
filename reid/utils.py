# -*- coding: utf-8 -*-
'''
    reference:
'''

import torch
from torch.distributions import laplace
from torch.distributions import uniform
import numpy as np
import os
from sklearn.metrics import average_precision_score
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import sys
import json

device = torch.device('cuda')

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)

def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    # TODO: Currently only considered one way of "uniform" sampling
    # for Linf, there are 3 ways:
    #   1) true uniform sampling by first calculate the rectangle then sample
    #   2) uniform in eps box then truncate using data domain (implemented)
    #   3) uniform sample in data domain then truncate with eps box
    # for L2, true uniform sampling is hard, since it requires uniform sampling
    #   inside a intersection of cube and ball, so there are 2 ways:
    #   1) uniform sample in the data domain, then truncate using the L2 ball
    #       (implemented)
    #   2) uniform sample in the L2 ball, then truncate using the data domain
    # for L1: uniform l1 ball init, then truncate using the data domain

    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(clip_min, clip_max)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    elif ord == 1:
        ini = laplace.Laplace(
            loc=delta.new_tensor(0), scale=delta.new_tensor(1))
        delta.data = ini.sample(delta.data.shape)
        delta.data = normalize_by_pnorm(delta.data, p=1)
        ray = uniform.Uniform(0, eps).sample()
        delta.data *= ray
        delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data

def mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams):
    #distmat = distmat.cpu().numpy()
    m, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    aps = []
    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]

        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

# def cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100):
#     #distmat = distmat.cpu().numpy()
#     m, _ = distmat.shape

#     indices = np.argsort(distmat, axis=1)
#     matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

#     ret = np.zeros(topk)
#     num_valid_queries = 0
#     for i in range(m):
#         valid = ((gallery_ids[indices[i]] != query_ids[i]) |
#                 (gallery_cams[indices[i]] != query_cams[i]))

#         y_true = matches[i, valid]
#         if not np.any(y_true): continue
#         index = np.nonzero(y_true)[0]
#         if index.flatten()[0] < topk:
#             ret[index.flatten()[0]] += 1
#         num_valid_queries += 1
#     if num_valid_queries == 0:
#         raise RuntimeError("No valid query")

#     return ret.cumsum() / num_valid_queries

def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    # distmat = distmat.cpu().numpy()
    def _unique_sample(ids_dict, num):
        mask = np.zeros(num, dtype=np.bool)
        for _, indices in ids_dict.items():
            i = np.random.choice(indices)
            mask[i] = True
        return mask

    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tensor2img(tensors, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
    for c in range(3):
        tensors.data[:, c, :, :] = tensors.data[:, c, :, :] * std[c] + mean[c]



def get_gallery_feats(model, gallery_loader):
    gallery_features = OrderedDict()
    gallery_labels = OrderedDict()
    for _, inputs in tqdm(enumerate(gallery_loader), desc='Extract gallery features..'):
        imgs, vids, fnames = inputs[0].to(device), inputs[1].to(device), inputs[-1]
        model.eval()
        feats = model(imgs, is_training=False).data.cpu()
        for fname, feat, vid in zip(fnames, feats, vids):
            gallery_features[fname] = feat
            gallery_labels[fname] = vid.item()

    return gallery_features, gallery_labels

def get_query_feats(model, query_loader):
    query_features = OrderedDict()
    # query_labels = OrderedDict()
    for _, inputs in tqdm(enumerate(query_loader), desc='Extract query features...'):
        imgs, fnames = inputs[0].to(device), inputs[-1]
        model.eval()
        feats = model(imgs, is_training=False).data.cpu()
        for fname, feat in zip(fnames, feats):
            query_features[fname] = feat

    # query_features = torch.stack([query_features[q[-1]] for q in query])
    return query_features

def distance_matrix(x, y):
    m,n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    dist = dist.cpu().numpy()

    return dist

def cal_mAP_cmc(query_features, gallery_features, query, gallery, dataset):
    x = torch.stack([query_features[q[-1]] for q in query])
    # x = query_features
    y = torch.stack([gallery_features[g[-1]] for g in gallery])

    dist = distance_matrix(x, y)

    query_ids = np.asarray([q[1] for q in query])
    gallery_ids = np.asarray([q[1] for q in gallery])
    query_cams = np.asarray([q[2] for q in query])
    gallery_cams = np.asarray([q[2] for q in gallery])

    mAP = mean_ap(dist, query_ids, gallery_ids, query_cams, gallery_cams)
    cmc_configs = {
        'CUHK03': dict(separate_camera_set=True,
                        single_gallery_shot=True,
                        first_match_break=False),
        'Market1501': dict(separate_camera_set=False,
                            single_gallery_shot=False,
                            first_match_break=True)
    }
    ranks = cmc(dist, query_ids, gallery_ids, query_cams, gallery_cams, **cmc_configs[dataset])
    return mAP, ranks


class Logger(object):
    def __init__(self, log_dir):
        self.file = open(os.path.join(log_dir, 'log.txt'), 'w')
        self.console = sys.stdout

    # def __del__(self):
    #     self.close()

    def __enter__(self):
        pass


    # def __exit__(self, *args):
    #     self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file:
            self.file.write(msg)
            self.flush()

    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())

    # def close(self):
    #     self.console.close()
    #     if self.file:
    #         self.file.close()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
