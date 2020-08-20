import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from pytorch_pretrained_bert import BertTokenizer
#import stanfordnlp

import os
from glob import glob
import random
import pickle
import json
import numpy as np
#import scipy.io
from PIL import Image, ImageDraw
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# -- local import ----------------
#from utils import get_iou, get_include
base_dir = 'dataset/LSMDC/'

gender_dir = base_dir + 'supplementary/csv_ids'
im_dir = base_dir + 'features_mvad/image/'
hb_dir = base_dir + 'features_mvad/human_bbox/'
hb_agg_dir = base_dir + 'features_mvad/human_pose_agg/'
i3d_dir = base_dir + 'features_mvad/i3d_rgb_map/'
i3d_agg_dir = base_dir + 'features_mvad/human_i3d/'
face_dir = base_dir + 'features_mvad/human_head/'
hb_name_idr = base_dir + 'features_mvad/human_id/'

mvad_pkl = {'train':base_dir + 'features_mvad/MVAD_train_agg.pkl',
            'val':base_dir + 'features_mvad/MVAD_val_agg.pkl',
            'test':base_dir + 'features_mvad/MVAD_test_agg.pkl'}


class MVADDataset(Dataset):

    def __init__(self, mode='train', bert_model='bert-base-uncased',
                 max_sent_len=40, hN=8, feats=['i3d_rgb','face']):
        '''
          params:
            mode: train, val
            max_sent_len: int
            iou_threshold: float
            neg_sample_num: int
        '''
        self.bert_range = 'whole'
        self.mode = mode
        self.feats = feats
        self.hN = hN # sample image number per clip
        # --- Load MPII data --------------------------
        with open(mvad_pkl[mode], 'rb') as f:
            mvad_data = pickle.load(f)

        # --- Load Bert ---------------------------------
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_cls_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.bert_sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.bert_msk_id = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

        for agg5 in tqdm(mvad_data, ncols=60):
            for clip_data in agg5:
                clip_id = clip_data['clip_id']
                word_list = clip_data['word_list']
                # --- Tokenize sentence -----------------------
                sent_emb = []
                word2tok = []
                for si, word in enumerate(word_list):
                    if len(word) > 0:
                        word2tok.append(len(sent_emb))
                        sent_emb += self.tokenize(word)
                clip_data['bert'] = sent_emb
                # Change word index into tok index
                for someone_info in clip_data['someone_info']:
                    someone_info['bert_loc'] = word2tok[someone_info['loc']]
                    try:
                        someone_info['bert_dep'] = word2tok[someone_info['dep']]
                    except:
                        someone_info['bert_dep'] = word2tok[-2]

        self.datas = mvad_data
        print("\# of total_data: ", len(self.datas))

    def tokenize(self, text, max_len=512):
        tokenized_text = self.tokenizer.tokenize(text)[:max_len]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_tokens

    def __getitem__(self, idx):
        '''
        idx:
          5 x [clip_id, someone_locs, sent_emb]
        return
          sent_emb : sN x [L]
          sent_tok : sN x [L]
          sent_msk : sN x [L]
          some_msk : sN x [L]
          clip_msk : sN x [5]
          im       : [5 x hN x 3 x 224 x 224]
          im_msk   : [5 x hN]
          pos_5    : [5 x hN x 17 x 2]
          pos_msk_5: [5 x hN x 17]
          meta     : sN x {'clip_id', 's_idx', 'sent'}
        '''
        # Build Bert
        agg5 = self.datas[idx]
        L = sum([len(cd['bert']) for cd in agg5]) + 6
        sN = sum([len(cd['someone_info']) for cd in agg5])
        sent_emb_t = np.zeros((1, L), dtype=np.int64)
        sent_tok_t = np.zeros((1, L), dtype=np.int64)
        sent_msk_t = np.ones((1, L), dtype=np.int64)
        some_msk_t = np.zeros((sN, L))
        dep_root_msk_t = np.zeros((sN, L))
        clip_msk_t = np.zeros((sN, 5))
        gt_position_t = np.zeros((sN, self.hN))
        gt_mat_t   = np.zeros((sN, sN))
        gt_gender_t = np.zeros((sN, 2))  # 0: logit, 1: mask
        meta_t, clip_info, s_pids= [], [], []
        s_id, cur_len = 0, 1
        sent_emb_t[0,0] = self.bert_cls_id
        for c_idx, cd in enumerate(agg5):
            # ----- sent processing --------------------------
            sent_emb_t[0, cur_len:cur_len + len(cd['bert'])] = cd['bert']
            sent_emb_t[0, cur_len + len(cd['bert'])] = self.bert_sep_id
            pos_trjs, pos_trjs_names, trjs = [], [], {}
            for s_idx, sd in enumerate(cd['someone_info']):
                some_msk_t[s_id, sd['bert_loc'] + cur_len] = 1
                dep_root_msk_t[s_id, sd['bert_dep'] + cur_len] = 1
                clip_msk_t[s_id, c_idx] = 1.
                trj_id = str(sd['hb_id'])
                if trj_id not in trjs:
                    trjs[trj_id] = s_idx
                    pos_trjs.append(trj_id)
                gt_position_t[s_id, trjs[trj_id]] = 1.
                # gender
                if sd['gender'] != 2:
                    gt_gender_t[s_id][0] = sd['gender']
                    gt_gender_t[s_id][1] = 1
                s_pids.append(sd['name'])
                pos_trjs_names.append(sd['name'])
                meta_t.append({'clip_id': cd['clip_id'],
                               'c_idx': c_idx,
                               's_idx': s_idx,
                               't_gt_idx': trjs[str(trj_id)],
                               'sent': cd['sent']})
                s_id += 1
            cur_len += len(cd['bert']) + 1
            clip_info.append((cd['clip_id'], pos_trjs, pos_trjs_names))
        # Text re-id GT
        for i in range(sN):
            for j in range(sN):
                if s_pids[i] == s_pids[j]:
                    gt_mat_t[i][j] = 1.

        # Build Visual
        hN = self.hN
        im_5, im_msk_5, bbox_meta_5 = [], np.zeros((5,hN)), np.zeros((5,hN,4))
        pos_5, pos_msk_5 = np.zeros((5,hN,17,2)), np.zeros((5,hN,17))
        i3d_rgb_5, i3d_flow_5 = np.zeros((5,hN,1024)), np.zeros((5, hN, 1024))
        face_5, face_msk_5 = np.zeros((5,hN,512)), np.zeros((5,hN))
        gt_vmat, gt_vmat_msk = np.zeros((5*hN, 5*hN)), np.zeros((5*hN, 5*hN))
        vid_list = []
        im_cache, hb_agg_cache, i3d_cache, face_cache = {}, {}, {}, {}

        # Load track info
        for c_idx, (clip_id, pos_trjs, pos_trjs_names) in enumerate(clip_info):
            # ----- image processing -------------------------
            # Sample at most hN hbbox.
            mov_id = clip_id.rsplit('_',1)[0]
            hb_agg_file = os.path.join(hb_agg_dir, mov_id, clip_id, 'best_pose.pkl')
            if hb_agg_file not in hb_agg_cache:
                with open(hb_agg_file, 'rb') as f:
                    hb_agg_cache[hb_agg_file] = pickle.load(f)
            # Load hb_name
            hb_name_file = os.path.join(hb_name_idr, mov_id, clip_id, 'bbox_id.pkl')
            with open(hb_name_file, 'rb') as f:
                hb_name = pickle.load(f) # hbid to name
            # Use i3d agg
            i3d_file = os.path.join(i3d_agg_dir, mov_id, clip_id ,'i3d_rgb.pkl')
            if i3d_file not in i3d_cache:
                with open(i3d_file, 'rb') as f:
                    i3d_cache[i3d_file] = pickle.load(f) # [T x 7 x 7 x 1024]
            vid_list += pos_trjs_names
            samps = pos_trjs

            if len(pos_trjs) < hN:
                neg_cands = []
                for hb_id in hb_agg_cache[hb_agg_file]:
                    if hb_id in hb_name:
                        if hb_id not in pos_trjs:
                            if hb_name[hb_id] not in pos_trjs_names:
                                neg_cands.append((hb_name[hb_id], hb_id))
                neg_num = hN - len(pos_trjs)
                if len(neg_cands) > neg_num:
                    if self.mode == 'train':
                        neg_cands = random.sample(neg_cands, k=neg_num)
                    else: #sort
                        def get_traj_len(hb_id):
                            if hb_id in i3d_cache[i3d_file]:
                                return i3d_cache[i3d_file][hb_id]['len']
                            return 0
                        neg_cands.sort(key = lambda hb_id: -1 * get_traj_len(hb_id))
                        neg_cands = neg_cands[:neg_num]
                for neg_cand in neg_cands:
                    vid_list.append(neg_cand[0])
                    samps.append(neg_cand[1])
            if len(samps) < hN:
                for _ in range(hN - len(samps)):
                    vid_list.append('[UNK]')
            im_clip = []
            for s_idx, hb_id in enumerate(samps):
                if hb_id not in hb_agg_cache[hb_agg_file]:
                    #print("hb_id missing", hb_id, hb_agg_file)
                    im_clip.append(torch.zeros([3,224,224]))
                    continue
                hb_data = hb_agg_cache[hb_agg_file][hb_id]
                # ---- Load Pose ------------------------------------------------------
                pos = hb_data['pos']
                pos_5[c_idx, s_idx] = pos[:,:2]
                pos_msk_5[c_idx, s_idx] = np.where(pos[:,2] > 0.3 , 1., 0.)
                x1, y1, x2, y2 = hb_data['hbbox']

                # ---- Load Image -----------------------------------------------------
                im_file = os.path.join(im_dir, mov_id, clip_id, hb_data['frame'] + '.jpg')
                if im_file not in im_cache:
                    im_cache[im_dir] = Image.open(im_file)
                img = im_cache[im_dir]
                im_w, im_h = img.size
                # Crop and resize
                img = torchvision.transforms.functional.crop(img, int(y1), int(x1),
                            int(y2-y1), int(x2-x1))
                img = torchvision.transforms.functional.resize(img, (224,224)) # [224 x 224 x 3]
                img = torchvision.transforms.functional.to_tensor(img) # [3 x 224 x 224]
                img = torchvision.transforms.functional.normalize(img,
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                im_clip.append(img)
                im_msk_5[c_idx, s_idx] = 1.

                # ---- Load bbox meta data
                mx = ((x1 + x2) / 2 )/ im_w
                my = ((y1 + y2) / 2 )/ im_h
                area = (x2 - x1) * (y2 - y1) / (im_w * im_h)
                bbox_meta_5[c_idx, s_idx, :3] = (mx, my, area)

                # ---- Load i3d --------------------------------------------------------
                if "i3d_rgb" in self.feats:
                    i3d_file = os.path.join(i3d_agg_dir, mov_id, clip_id ,'i3d_rgb.pkl')
                    if i3d_file not in i3d_cache:
                        with open(i3d_file, 'rb') as f:
                            i3d_cache[i3d_file] = pickle.load(f) # [T x 7 x 7 x 1024]
                    if hb_id in i3d_cache[i3d_file]:
                        i3d_feat = i3d_cache[i3d_file][hb_id]['feat']
                        i3d_length = i3d_cache[i3d_file][hb_id]['len']
                        i3d_rgb_5[c_idx, s_idx] = i3d_feat
                        bbox_meta_5[c_idx, s_idx, 3] = np.clip(i3d_length/300, 0, 1)

                # ----- Load face ---------------------------------------------------------
                if 'face' in self.feats:
                    face_file = os.path.join(face_dir, mov_id, clip_id, 'head_feature.pkl')
                    if face_file not in face_cache:
                        with open(face_file, 'rb') as f:
                            face_cache[face_file] = pickle.load(f)
                    if int(hb_id) in face_cache[face_file]:
                        face_feat = face_cache[face_file][int(hb_id)]
                        if len(face_feat) != 0: # If no face, feature is empty list
                            face_5[c_idx, s_idx] = face_feat[0]
                            face_msk_5[c_idx, s_idx] = 1.
                    else:
                        print("no hbid in face", clip_id, hb_id)
            # Pad
            if len(samps) < hN:
                for _ in range(hN - len(samps)):
                    im_clip.append(torch.zeros([3,224,224]))
            im_5.append(torch.stack(im_clip, 0)) # [4 x 3 x 224 x 224]
        im_5 = torch.stack(im_5, 0) # [5 x 4 x 3 x 224 x 224]
        # build gt_vmat
        for i in range(5*hN):
            for j in range(5*hN):
                if i == j:
                    continue
                if vid_list[i] != '[UNK]' and vid_list[j] != '[UNK]':
                    gt_vmat_msk[i][j] = 1
                    if vid_list[i] == vid_list[j]:
                        gt_vmat[i][j] = 1
        return (sent_emb_t, sent_tok_t, sent_msk_t, some_msk_t, clip_msk_t,
                 im_5, im_msk_5, pos_5, pos_msk_5,
                 i3d_rgb_5, face_5, face_msk_5,
                 bbox_meta_5, dep_root_msk_t,
                 gt_gender_t, gt_position_t, gt_mat_t, gt_vmat, gt_vmat_msk, meta_t)

    def __len__(self):
        return len(self.datas)

def pad_collate(data):
    def pad_sequence(sequences, dtype='long'):
        B = len(sequences)
        sN = max([seq.shape[0] for seq in sequences])
        L = max([seq.shape[1] for seq in sequences])
        padded_seqs = torch.zeros((B, sN, L), dtype = getattr(torch,dtype))
        for bi, seq in enumerate(sequences):
            padded_seqs[bi, :seq.shape[0], :seq.shape[1]] = torch.tensor(seq)
        return padded_seqs
    column_data = list(zip(*data))
    all_values = []
    # embs, toks, msk
    for i in range(3):
        all_values.append(pad_sequence(column_data[i]))
    # some_msk, clip_msk
    for i in range(3,5):
        all_values.append(pad_sequence(column_data[i], dtype='float'))
    # im, im_msks
    all_values.append(torch.stack(column_data[5],0))
    # im_msks, pos, pos_msks, i3d_rgb, face, face_msks, bbox_meta_5,
    for i in range(6,13):
        all_values.append(torch.FloatTensor(column_data[i]))
    # dep_root_msk_t, gt_gender_t, gt_position_t, gt_mats, gt_vmat, gt_vmat_msk,
    for i in range(13,19):
        all_values.append(pad_sequence(column_data[i], dtype='float'))
    # meta datas
    all_values.append(column_data[19])
    return all_values

def get_dataloader(batch_size = 1, num_workers = 4,sample_num=4,feats=['i3d_rgb']):
    train_dset = MVADDataset(mode='train', hN=sample_num, feats=feats)
    train_dloader = DataLoader(train_dset, batch_size = batch_size,
                               num_workers = num_workers,
                               collate_fn = pad_collate,
                               shuffle = True)
    val_dset = MVADDataset(mode='val', hN=sample_num, feats=feats)
    val_dloader = DataLoader(val_dset, batch_size = batch_size,
                               num_workers = num_workers,
                               collate_fn = pad_collate,
                               shuffle = False)

    return train_dloader, val_dloader

if __name__ == "__main__":

    val_dset = MVADDataset(mode='val',hN = 8)
    val_dloader = DataLoader(val_dset, batch_size = 1,
                               num_workers = 0,
                               collate_fn = pad_collate,
                               shuffle = False)

    for res in tqdm(val_dloader):
        a = res

