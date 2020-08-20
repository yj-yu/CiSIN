import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_include(b1, b2):
    """
    Calculate the percentage of b1 (small) in b2 (big)
    """
    bb1 = {'x1': b1[0], 'y1': b1[1], 'x2': b1[2], 'y2': b1[3]}
    bb2 = {'x1': b2[0], 'y1': b2[1], 'x2': b2[2], 'y2': b2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    return intersection_area / min(bb1_area, bb2_area) 

def get_iou(b1, b2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : list of ['x1', 'y1', 'x2', 'y2']
    bb2 : list of ['x1', 'y1', 'x2', 'y2']
    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {'x1': b1[0], 'y1': b1[1], 'x2': b1[2], 'y2': b1[3]}
    bb2 = {'x1': b2[0], 'y1': b2[1], 'x2': b2[2], 'y2': b2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def masked_softmax(inp, mask):
    '''
    args
      inp: B x C x D
      mask: B x C x D (1: value, 0: mask)
    return 
      att: B x C x D (softmax in -1 axis)
    '''
    mask = 1 - mask
    inp.data.masked_fill_(mask.data.bool(), -float("inf"))
    att = F.softmax(inp, dim = -1)
    att.data.masked_fill_(att.data != att.data, 0)  # remove nan from softmax on -inf
    return att


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, num_frames, add_batch_norm=True, length_normalize=False):
        super(NetVLAD, self).__init__()
        self.num_frames = num_frames
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size))
                                     * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1 / math.sqrt(feature_size))
                                      * torch.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size
        self.length_normalize=length_normalize

    def set_num_frames(self, num):
        self.num_frames = num

    def forward(self, x):
        assignment = torch.matmul(x, self.clusters)

        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, self.num_frames, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, self.num_frames, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        if self.length_normalize:
            vlad = vlad / self.num_frames

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.view(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad
    
def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))