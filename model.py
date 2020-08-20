#v7 just sum not gated
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from torch.nn.parameter import Parameter

from pytorch_pretrained_bert import BertModel

from utils import masked_softmax, gelu
from model_sub import VisualReIdModel

class CISINModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', 
                 v_feats=[], t_feats=[], char_reid=[], use_gender=True,
                 hidden_dim=1024, dropout_prob=0.3):
        super(CISINModel, self).__init__()
        self.v_feats = v_feats
        self.t_feats = t_feats
        self.char_reid = char_reid
        self.use_gender = use_gender
        self.hidden_dim = hidden_dim

        ''' Model_type
         0: [S,A]-[F,I,M]
         1: [S,A]-[F] + [S,A]-[I] + [M]
         2: [S]-[F] + [A]-[I] + [M]
         3: [[s]-[f], [A]-[I], M] -> fusion
        '''
        # Text Embedding   
        self.bert = BertModel.from_pretrained(bert_model)
        s_dim = 768 * len(t_feats)
        # Action
        self.s_a_proj = nn.Sequential(
            nn.Linear(s_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        # Someone
        self.s_s_proj = nn.Sequential(
            nn.Linear(s_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
            
        # Text Re-Identification
        if 'text' in self.char_reid:
            assert('someone' in self.t_feats)
            self.text_reid_fc = nn.Sequential(
                nn.Linear(768, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Gender classifier
        if self.use_gender:
            assert('someone' in self.t_feats)
            self.gender_fc = nn.Sequential(
                nn.Linear(768, 768),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(768, 1),
                nn.Sigmoid()
            )

        # Visual embedding
        self.act_conv = torchvision.models.resnet50(pretrained=True)
        res_ftrs = self.act_conv.fc.in_features
        self.act_conv.fc = nn.Identity()
        v_dim = 0
        if 'img' in self.v_feats:
            v_dim += res_ftrs
        if 'i3d_rgb' in self.v_feats:
            v_dim += 1024
            
        # Visual projection
        self.i_a_proj = nn.Sequential(
            nn.Linear(v_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        self.i_s_proj = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh()
        )
        if 'meta' in self.v_feats:            
            self.meta_proj = nn.Sequential(
                nn.Linear(4, 50),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout_prob),
            )
        
        self.fuse_fc = nn.Sequential(
            nn.Linear(self.hidden_dim*2+50, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1) 
        )

        # Character Re-Identification
        if 'visual' in self.char_reid:
            score_model_feats = []
            if 'img_v' in self.v_feats:
                score_model_feats.append('img')
            if 'pose_v' in self.v_feats:
                score_model_feats.append('pose')
            if 'face_v' in self.v_feats:
                score_model_feats.append('face')
            self.vid_model = VisualReIdModel(feats = score_model_feats)

    def forward(self, embs, toks, msks, s_msks, c_msks, dep_root_msk, imgs, i_msks, poses, pose_msks, i3d_rgb=None, face=None, face_msks=None, bbox_meta=None):
        '''
        params:
          embs: [B x 1 x L]
          toks: [B x 1 x L]
          msks: [B x 1 x L]
          s_msks: [B x sN x L]
          c_msks: [B x sN x cN]
          dep_root_msk: [B x sN x L]
          imgs: [B x cN x hN x 3 x 224 x 224]
          i_msks: [B x cN x hN]
          poses: [B x cN x hN x 17 x 2]
          pose_msks: [B x cN x hN x 17]
          i3d_rgb: [B x cN x hN x 1024]
          face: [B x cN x hN x 512]
          face_msks: [B x cN x hN]
          bbox_meta_5: [B x cN x hN x 3]
        '''
        B, sN, L = s_msks.shape
        cN, hN = imgs.size(1), imgs.size(2)
        # Text embedding
        embs, toks, msks = embs.view(-1, L), toks.view(-1, L), msks.view(-1,L) # [B*sN x L]
        bert_x, _ = self.bert(embs, toks, msks, output_all_encoded_layers=False) # [B x L x 768]
        bert_x = gelu(bert_x).view(B, 1, L, 768)
        if 'someone' in self.t_feats:
            ts_x = bert_x * s_msks.view(B, sN, L, 1) # [B x sN x L x 768]
            ts_x = torch.sum(ts_x, 2) # [B x sN x 768]
        if 'action' in self.t_feats:
            ta_x = bert_x * dep_root_msk.view(B, sN, L, 1) # [B x sN x L x 768]
            ta_x = torch.sum(ta_x, 2) # [B x sN x 768]
        tce_x = torch.cat([ta_x, ts_x], -1) # [B x sN x sdim]
        
        # Text Projection
        s_a_x = self.s_a_proj(tce_x) # [B x sN x H]
        s_s_x = self.s_s_proj(tce_x) # [B x sN x H]
        
        # Auxiliary Gender Classifier
        if self.use_gender:
            g_x = self.gender_fc(ts_x) # [B x sN x 1]
            self.gender_result = g_x.squeeze(-1) # [B x sN]
        
        # Visual embedding
        i_x = torch.zeros((B*cN*hN, 0), device=bbox_meta.device)
        if 'img' in self.v_feats:
            imgs = imgs.view(-1, 3, 224, 224)
            img_x = self.act_conv(imgs) # [B*cN*hN x 2048]
            i_x = torch.cat((i_x, img_x), -1) #[B*cN*hN x (2048 + 4)]
        if "i3d_rgb" in self.v_feats:
            i3d_x = i3d_rgb.view(-1, 1024) # [B*cN*hN x 1024]
            i_x = torch.cat((i_x, i3d_x), -1) # [B*cN*hN x num_ftrs]
        if 'face' in self.v_feats:
            face_x = face.view(-1, 512) # [B*cN*hN x 512]
            
        # Image projection
        i_a_x = self.i_a_proj(i_x) # [B*cN*hN x H]
        i_a_x = i_a_x.view((B, cN*hN, -1)) # [B x cN*hN x H]
        i_s_x = self.i_s_proj(face_x) # [B*cN*hN x H]
        i_s_x = i_s_x.view((B, cN*hN, -1)) # [B x cN*hN x H]
        if 'meta' in self.v_feats:
            meta_x = self.meta_proj(bbox_meta.view(-1,4)) # [B*cN*hN x 50]
        
        # Character Grounding
        s_a_x = s_a_x.unsqueeze(2).repeat(1,1,cN*hN,1).view(-1,self.hidden_dim)
        i_a_x = i_a_x.unsqueeze(1).repeat(1,sN,1,1).view(-1,self.hidden_dim)
        s_s_x = s_s_x.unsqueeze(2).repeat(1,1,cN*hN,1).view(-1,self.hidden_dim)
        i_s_x = i_s_x.unsqueeze(1).repeat(1,sN,1,1).view(-1,self.hidden_dim)
        if 'meta' in self.v_feats:
            meta_x = meta_x.view(B, 1, cN*hN, 50).repeat(1,sN,1,1) # B x sN x cN*hN x 50
                                         
        f_a_x = s_a_x * i_a_x
        f_a_x = f_a_x.view(B, sN, cN*hN, self.hidden_dim) # [B x sN x cN*hN x H]
        f_s_x = s_s_x * i_s_x
        f_s_x = f_s_x.view(B, sN, cN*hN, self.hidden_dim) # [B x sN x cN*hN x H]
        f_t_x = torch.cat((meta_x, f_a_x, f_s_x),-1) # [B x sN x cN*hN x fH]
        f_x = self.fuse_fc(f_t_x) # [B x sN x cN*hN x 1]
        f_x = f_x.squeeze(-1) # [B x sN x cN*hN]
        
        # Masking
        pred_ground = f_x * i_msks.view(B, 1, cN*hN) * s_msks.sum(-1, keepdim=True)
        
        # Character Re-Identification
        pred_chreid = 0
        to_divide = 1
        self.text_mat, self.vid_mat, self.vgv_mat = None, None, None
         # Text Re-Id
        if 'text' in self.char_reid:
            text_reid_x = ts_x.unsqueeze(1) * ts_x.unsqueeze(2) # [B x sN x sN x 768]
            text_reid_x = self.text_reid_fc(text_reid_x)
            text_mat = text_reid_x.squeeze(-1) # [B x sN x sN]
            self.text_mat = text_mat
            pred_chreid += self.text_mat
            to_divide += 1
        # Person identity representation
        if 'visual' in self.char_reid:
            vid_mat = self.vid_model(imgs.view(B, -1, 3, 224, 224),
                                     i_msks.view(B, -1),
                                     poses.view(B, -1, 17, 2),
                                     pose_msks.view(B, -1, 17),
                                     face.view(B, -1, 512),
                                     face_msks.view(B, -1)) # B x cN*hN x cN*hN

            self.vid_mat = vid_mat
            p_msks = c_msks.view(B,sN,cN,1).repeat(1,1,1,hN).view(B,sN,cN*hN)
            i_mm = i_msks.view(B, 1, cN*hN) # [B x 1 x cN*hN]
            p_msks = p_msks * i_mm # [B x sN x cN*hN] (1 if exist, 0 if not)
            p_att = masked_softmax(f_x * 5, p_msks) #[B x sN x cN*hN], sharpen f_x little.
            
            vgv_mat = torch.matmul(p_att, vid_mat)
            vgv_mat = vgv_mat * (1 - p_msks) - 0.1 * p_msks # Mask for someone in same clip
            vgv_mat = torch.matmul(vgv_mat, p_att.transpose(1,2)) # [B x sN x sN]
            vgv_mat = torch.sigmoid(vgv_mat) # [B x sN x sN]  0 ~ 1
            self.vgv_mat = vgv_mat

            pred_chreid = (pred_chreid + vgv_mat) / to_divide

        return pred_ground, pred_chreid
