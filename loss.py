import torch
import torch.nn.functional as F
from utils import masked_softmax


def margin_loss(pos_scores, neg_scores, msks, margin=3):
    num = msks.sum(-1).clamp(min=1) # B
    m_loss = F.relu(margin + neg_scores - pos_scores) # [B x sN]
    m_loss = (m_loss * msks).sum(-1)/num # [B]
    return m_loss.mean() # [1]
    
def ce_loss(pred, gt, ground_msk, s_msks):
    s_num = s_msks.sum(-1).clamp(min=1)
    pred = masked_softmax(pred, ground_msk)
    m_loss = F.binary_cross_entropy(pred, gt, reduction='none') #[B x sN x cN*hN]
    m_loss = m_loss * ground_msk # CE loss
    m_loss = m_loss.mean(-1) #[B x sN]
    m_loss = (m_loss * s_msks).sum(-1)/ s_num # [B]
    m_loss = m_loss.mean() # [1]
    return m_loss

def reid_loss(pred_reid, b_gt_mats, s_msks):
    sN = s_msks.size(1)
    s_num = s_msks.sum(-1)
    c_loss = F.binary_cross_entropy(pred_reid, b_gt_mats, reduction='none') # [B x sN x sN]
    eye_mask = 1 - torch.eye(sN, device=s_msks.device) # [sN x sN]
    eye_mask = eye_mask.unsqueeze(0) # [1 x sN x sN]
    mat_mask = s_msks.unsqueeze(1) * s_msks.unsqueeze(2) * eye_mask
    c_loss = c_loss * mat_mask
    c_loss = c_loss.sum((1,2)) / (s_num*s_num - s_num).clamp(min=1)
    c_loss = c_loss.mean()
    return c_loss

def vreid_loss(pred_reid, b_gt_mats, msks):
    v_loss = F.binary_cross_entropy(pred_reid, b_gt_mats, reduction='none') # [B x sN x sN]
    v_loss = v_loss * msks # [B x sN x sN]
    v_loss = v_loss.sum((1,2)) / msks.sum((1,2)).clamp(min=1) # B
    v_loss = v_loss.mean()
    return v_loss

def gender_loss(gender_logit, gender_gt, gender_msk):
    #gender_logit : [B x sN]
    
    g_loss = F.binary_cross_entropy(gender_logit, gender_gt, reduction='none') # [B x sN]
    g_loss = g_loss * gender_msk
    g_loss = g_loss.sum() / gender_msk.sum(-1).clamp(min=1) # [B]
    g_loss = g_loss.mean() # [1]
    return g_loss