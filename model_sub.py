import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from pytorch_pretrained_bert import BertModel
from utils import masked_softmax

class VisualReIdModel(nn.Module):
    def __init__(self, feats,  score_layer=True):
        super(VisualReIdModel, self).__init__()
        self.feats = feats
        self.score_layer = score_layer
        num_ftrs = 2048
        self.hidden_dim = 1024
        self.dist = 'cos'
        
        self.pid_conv = torchvision.models.resnet50(pretrained=True)
        self.pid_conv.fc = nn.Identity()
        self.pid_conv.avgpool = nn.Identity()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        if self.score_layer:
            score_fc_list = []
            if 'img' in self.feats:
                #self.i_proj = nn.Linear(num_ftrs, self.hidden_dim)
                self.i_score_fc = nn.Linear(1, 1)
                score_fc_list.append(self.i_score_fc)
            if 'pose' in self.feats:
                #self.p_proj = nn.Linear(num_ftrs, self.hidden_dim)
                self.p_score_fc = nn.Linear(1, 1)
                score_fc_list.append(self.p_score_fc)
            if 'face' in self.feats:
                self.f_proj1 = nn.Linear(512, self.hidden_dim)
                self.f_proj2 = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.f_score_fc = nn.Linear(1, 1)
                score_fc_list.append(self.f_score_fc)
            for score_fc in score_fc_list:
                nn.init.constant_(score_fc.bias, -0.5)
                nn.init.constant_(score_fc.weight, 1.)

    def forward(self, imgs, i_msks, poses=None, pose_msks=None, face=None, face_msks=None):
        '''
        params:
          imgs: [B x hN x 3 x 224 x 224]
          i_msks: [B x hN]
          poses: [B x hN x 17 x 2]
          pose_msks: [B x hN x 17]
          face: [B x hN x 512]
          face_msks: [B x hN]
        return:
          score: [B x hN x hN] ( 0 ~ 1 )
         '''
        B, hN = i_msks.shape
        # Person identity representation
        imgs = imgs.view(-1, 3, 224, 224)
        #with torch.no_grad():
        i_2d_x = self.pid_conv(imgs) # [B*hN x 2048 x 14 x 14] (B, C, H, W)
        i_2d_x = i_2d_x.view(B*hN, 2048, 7,7)
        score = 0
        if 'img' in self.feats:
            i_x = self.avg_pool(i_2d_x).squeeze()
            i_x = i_x.view(B, hN, -1)#self.hidden_dim) # [B x hN x H]
            i_x = torch.nn.functional.normalize(i_x, dim=-1)
            i_score = torch.matmul(i_x, i_x.transpose(1,2)) # [B x hN x hN]
            if self.score_layer:
                i_score=self.i_score_fc(i_score.unsqueeze(-1)).squeeze(-1)
            score += i_score

        if 'pose' in self.feats:
            b_size = i_2d_x.size(0)
            poses = poses.view(-1, 17, 2)
            # gather x,y coordinated features
            x = (poses[:,:,1] * 6.99).long()
            y = (poses[:,:,0] * 6.99).long()
            p_x =  i_2d_x[torch.arange(b_size)[:, None], :, y, x] # [B*hN x 17 x 2048]
            # Projcet into H space
            #p_x = self.p_proj(p_x)
            p_x = p_x.view(B, hN, 17, -1)#self.hidden_dim) # [B x hN x 17 x H]
            p_x = torch.nn.functional.normalize(p_x, dim=-1)
            # Calculate score
            p_x = p_x * pose_msks.unsqueeze(-1) # [B x hN x 17 x H]
            p_x = p_x.view(B, hN, -1) # [B x hN x 17*H]
            p_score = torch.matmul(p_x, p_x.transpose(1,2)) # [B x hN x hN]
            # Since 0 x value = 0, each B, i, j is sum of pi*pj where pi and pj are both valid.
            normalize_msk = torch.matmul(pose_msks, pose_msks.transpose(1,2))
            p_score = p_score / (F.relu(normalize_msk - 1) + 1)
            if self.score_layer:
                p_score = self.p_score_fc(p_score.unsqueeze(-1)).squeeze(-1)
            score += p_score

        if 'face' in self.feats:
            #f_x = face#self.f_proj(face)
            f_x = self.f_proj1(face)
            f_x = F.leaky_relu(f_x)
            f_x = self.f_proj2(f_x)
            self.vte_face = f_x
            f_x = torch.nn.functional.normalize(f_x, dim=-1) # [B x hN x H]
            f_x = f_x * face_msks.unsqueeze(-1)
            f_score = torch.matmul(face, face.transpose(1,2))# [B x hN x hN] ###
            if self.score_layer:
                f_score = self.f_score_fc(f_score.unsqueeze(-1)).squeeze(-1)
            score += 2 * f_score
        # Mask
        score = score * i_msks.unsqueeze(1) * i_msks.unsqueeze(2)

        score_weight = (len(self.feats))
        if 'face' in self.feats:
            score_weight += 1
        score = score / score_weight#(len(self.feats))
        score = torch.tanh(score)
        
        return score

