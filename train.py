#from mpii_dataloader import MPIIDataset, mpii_pad_collate
from dataloader import get_dataloader
from model import CISINModel
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import json
import pickle

from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm

import loss
from eval_utils import calc_acc

def train(args, model, optimizer, train_dloader, val_dloader, summary):
    # ----------- Start training ----------------------------------
    best_eval_acc = 0
    start_epoch = 0
    iter_calc = 0
    for epoch in range(start_epoch, start_epoch + args['max_epoches']):
        # ------- Train ---------------------------
        model.train()
        titer = tqdm(train_dloader, ncols=60)
        results = defaultdict(list)
        for res in titer:
            iter_calc += 1
            res_cuda = [r.cuda() for r in res[:-1]]
            b_embs, b_toks, b_msks, b_s_msks, b_c_msks, b_imgs, b_im_msks, b_poses, b_pose_msks, b_i3d_rgb, b_face, b_face_msks, b_bbox_meta, b_dep_root_msk, b_gt_gender, b_gt_positions, b_gt_mats, b_gt_vmat, b_gt_vmat_msk = res_cuda
            B, sN, L = b_s_msks.shape
            cN, hN = b_imgs.size(1), b_imgs.size(2)
            pred_ground, pred_reid = model(b_embs, b_toks, b_msks, b_s_msks, b_c_msks, b_dep_root_msk, b_imgs, b_im_msks, b_poses, b_pose_msks, b_i3d_rgb, b_face, b_face_msks, b_bbox_meta) # B x sN x cN*hN
            # ---------- Get Loss ------------------------------------------
            # ---- Image wise ------------------------
            ground_clip_msk = b_c_msks.view(B,sN,cN,1).repeat(1,1,1,hN).view(B,sN,cN*hN)
            ground_msk = ground_clip_msk * b_im_msks.view(B, 1, cN*hN) # B x sN x cN*hN 
            s_msks = b_s_msks.sum(-1) # [B, sN]
            s_num = s_msks.sum(-1) # [B] number of sN in each batch

            gt_position = b_gt_positions.view(B,sN,1,hN).repeat(1,1,cN,1).view(B,sN,cN*hN)
            gt_position = ground_msk * gt_position # [B x sN x cN*hN], 1 if positive
            pos_msk = 1 - gt_position # 0 if positive   
            pos_scores, _ = (pred_ground - pos_msk * 20).max(-1) # [B x sN]
            neg_msk = (1 - ground_msk) + gt_position # 0 if in clip negative
            neg_scores, _ = (pred_ground - neg_msk * 20).max(-1) # [B x sN]
            if args['loss'] == 'margin':
                loss1 = loss.margin_loss(pos_scores, neg_scores, s_msks)
                # other way
                pos_scores2, _ = (pred_ground - pos_msk * 20).max(1) # [B x cN*hN]
                neg_msk2 = (1 - pos_msk * s_msks.unsqueeze(-1))
                neg_scores2, _ = (pred_ground - neg_msk2 * 20).max(1) # [B x cN*hN]
                loss2 = loss.margin_loss(pos_scores2, neg_scores2, gt_position.sum(1))
                total_loss = loss1 + loss2
                results['ground_loss'].append(total_loss.cpu().item())
            elif args['loss'] == 'ce':
                total_loss = loss.ce_loss(pred_ground, gt_position, ground_msk, s_msks)
                results['ground_loss'].append(total_loss.cpu().item())

            if args['char_reid']:
                c_loss = loss.reid_loss(pred_reid, b_gt_mats, s_msks)
                total_loss += 0.1 * c_loss
                results['reid_loss'].append(c_loss.cpu().item())
                
                v_loss = loss.vreid_loss(torch.sigmoid(model.vid_mat), b_gt_vmat, b_gt_vmat_msk)
                total_loss += 0.1 * v_loss
                results['vreid_loss'].append(v_loss.cpu().item())
            if args['use_gender']:
                gender_logit = model.gender_result
                gender_gt = b_gt_gender[:, :, 0]
                gender_msk = b_gt_gender[:, :, 1]
                g_loss = loss.gender_loss(gender_logit, gender_gt, gender_msk)            
                total_loss += g_loss
                results['gender_loss'].append(g_loss.cpu().item())

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # ---- Calculate acc --------------------------------------------
            _, preds = torch.stack([neg_scores, pos_scores], -1).max(-1)
            preds = preds.cpu() # [B x sN]
            for bidx, b_pred in enumerate(preds):
                for sidx in range(len(res[-1][bidx])):
                    if b_pred[sidx] == 1: #pos is larger
                        results['ground_acc'].append(1)
                    else:
                        results['ground_acc'].append(0)
            if args['use_gender']:
                g_preds = (abs(gender_logit - gender_gt)).cpu()
                gender_msk_cpu = gender_msk.cpu()
                for bidx, g_pred in enumerate(g_preds):
                    for sidx, gp in enumerate(g_pred):
                        if gender_msk_cpu[bidx, sidx] == 1:
                            if g_pred[sidx] < 0.5: #pos is larger
                                results['gender_acc'].append(1)
                            else:
                                results['gender_acc'].append(0)                
        #lr_scheduler.step()
        # ---- Print loss, acc -------------------------------------------------
        print_str = ''
        for key in results:
            res_avg = sum(results[key]) / len(results[key])
            summary.add_scalar('train/'+key, res_avg, epoch)
            print_str += key + ': %.3f '%res_avg
        print(args['save_dir'])
        print(print_str)
        result, eval_scores = evaluate(args, model, val_dloader, epoch)
        # Save result
        if best_eval_acc < eval_scores['grounding']:
            best_eval_acc = eval_scores['grounding']
            torch.save({'epoch': epoch + 1,
                        'iter_calc': iter_calc,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
                        os.path.join('logdir', args['save_dir'], 'best_ckpt.pth.tar'))
            with open(os.path.join('logdir', args['save_dir'], 'mvad_output.pkl'), 'wb') as f:
                pickle.dump(result, f)
        
def evaluate(args, model, val_dloader, epoch):
    model.eval()
    evaliter = tqdm(val_dloader, ncols=60)
    eval_accs = defaultdict(list)
    result = {}
    for ei, res in enumerate(evaliter):
        res_cuda = [r.cuda() for r in res[:-1]]
        b_embs, b_toks, b_msks, b_s_msks, b_c_msks, b_imgs, b_im_msks, b_poses, b_pose_msks, b_i3d_rgb, b_face, b_face_msks, b_bbox_meta, b_dep_root_msk, b_gt_gender, b_gt_positions, b_gt_mats, b_gt_vmat, b_gt_vmat_msk = res_cuda
        B, sN, L = b_s_msks.shape
        cN, hN = b_imgs.size(1), b_imgs.size(2)
        with torch.no_grad():
            pred_ground, pred_reid = model(b_embs, b_toks, b_msks, b_s_msks, b_c_msks, b_dep_root_msk, b_imgs, b_im_msks, b_poses, b_pose_msks, b_i3d_rgb, b_face, b_face_msks, b_bbox_meta) # B x sN x cN*hN
            f_msk = b_c_msks.view(B,sN,cN,1).repeat(1,1,1,hN).view(B,sN,cN*hN)
            f_msk = f_msk * b_im_msks.view(B, 1, cN*hN)
            gt_msk = b_gt_positions.view(B,sN,1,hN).repeat(1,1,cN,1).view(B,sN,cN*hN)
            gt_f_msk = 1 - f_msk * gt_msk # [B x sN x cN*hN]
            f_pos, _ = (pred_ground - gt_f_msk * 20).max(-1) # [B x sN]
            neg_msk = (1 - f_msk * (1-gt_msk))
            f_neg, _ = (pred_ground - neg_msk * 20).max(-1) # [B x sN]
            _, preds = torch.stack([f_neg, f_pos],-1).max(-1)

        preds = preds.cpu().tolist() # [B x sN]
        for bidx, b_pred in enumerate(preds):
            for sidx in range(len(res[-1][bidx])):
                if b_pred[sidx] == 1: #pos is larger
                    eval_accs['grounding'].append(1)
                else:
                    eval_accs['grounding'].append(0)
        if args['use_gender']:
            g_preds = (abs(model.gender_result - b_gt_gender[:,:,0])).cpu()
            gender_msk_cpu = b_gt_gender[:,:,1].cpu()
            for bidx, g_pred in enumerate(g_preds):
                for sidx in range(len(res[-1][bidx])):
                    if gender_msk_cpu[bidx, sidx] == 1:
                        if g_pred[sidx] < 0.5: #pos is larger
                            eval_accs['gender'].append(1)
                        else:
                            eval_accs['gender'].append(0)
        # Save result
        ss_pred = pred_reid.cpu().numpy()
        ss_gt = b_gt_mats.cpu().numpy()
        f_x_cpu = (pred_ground).cpu().numpy()
        f_pos_ = f_pos.cpu().numpy()
        f_neg_ = f_neg.cpu().numpy()
        if model.vgv_mat is not None:
            vgv_mat_ = model.vgv_mat.cpu().numpy()
        if model.text_mat is not None:
            text_mat_ = model.text_mat.cpu().numpy()
        for b_idx, meta in enumerate(res[-1]):
            clip_id = meta[0]['clip_id']
            st_gt = [(m['c_idx'], m['s_idx']) for m in meta]
            sN, hN = len(meta), args['sample_num']
            st_pred = np.zeros((sN, hN))
            for mi, (ci, si) in enumerate(st_gt):
                st_pred[mi] = f_x_cpu[b_idx][mi, ci * hN: (ci + 1) * hN]
            result[clip_id] = {'char_pred': ss_pred[b_idx][:sN,:sN], 
                               'ground_pred': st_pred}
            if model.vgv_mat is not None:
                result[clip_id]['vgv_mat'] = vgv_mat_[b_idx][:sN,:sN]
            if model.text_mat is not None:
                result[clip_id]['text_mat_'] = text_mat_[b_idx][:sN,:sN]

            eval_accs['char_score'].append(calc_acc(sN, ss_gt[b_idx], ss_pred[b_idx]))
            if model.vgv_mat is not None:
                eval_accs['vgv_score'].append(calc_acc(sN, ss_gt[b_idx], vgv_mat_[b_idx]))
            if model.text_mat is not None:
                eval_accs['text_score'].append(calc_acc(sN, ss_gt[b_idx], text_mat_[b_idx]))
                
    eval_scores = {key: sum(eval_accs[key]) / len(eval_accs[key]) for key in eval_accs}
    eval_res_str = ' '.join([key + ": " + "%.4f"%eval_scores[key] for key in eval_scores])
    print("epoch: ", epoch, eval_res_str)
    summary.add_scalar('eval/char_reid',  eval_scores['char_score'], epoch)
    summary.add_scalar('eval/grounding', eval_scores['grounding'], epoch)
    with open(os.path.join('logdir', args['save_dir'], 'res_out.txt'), 'a') as f:
        f.write("epoch: " + str(epoch) + eval_res_str + "\n")
    
    return result, eval_scores
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='cisin_vtrain')
    opt = parser.parse_args()
    # Config
    args = {'seed':2020,
            'max_sent_len': 40,
            'sample_num': 8,
            'lr': 1e-5,
            'weight_decay':1e-7,
            'lr_decay': 0.99,
            'max_epoches': 40,
            'batch_size': 2,
            'bert_model': 'bert-base-uncased',
            'v_feats': ['i3d_rgb','face','meta', 'pose_v', 'face_v'], #['img','i3d_rgb',i3d_flow','face','meta', 'img_v','pose_v','face_v'],
            't_feats' : ['someone', 'action'], #['someone', 'action']
            'char_reid': ['text', 'visual'],# sent, img, sentonly
            'use_gender': True,
            'hidden_dim': 1024,
            'dropout_prob': 0.3,
            'loss': 'margin', # ['margin', 'ce'] 
            'load_path': opt.load_path,
            'save_dir': opt.save_dir
           }
    
    return args
            
if __name__ == '__main__':
    args = parse_args()
    
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    # -------------- Load dataset -----------------------
    print('Load dataset')
    train_dloader, val_dloader = get_dataloader(batch_size = args['batch_size'],
                                                num_workers = 8,
                                                sample_num = args['sample_num'],
                                                feats = args['v_feats'])

    # ---------- Set model & Optimizer -------------------------
    print('Load Model')
    model = CISINModel(args['bert_model'],
                       v_feats = args['v_feats'],
                       t_feats = args['t_feats'],                   
                       char_reid = args['char_reid'],
                       use_gender = args['use_gender'],
                       hidden_dim = args['hidden_dim'],
                       dropout_prob = args['dropout_prob'])

    model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr = args['lr'],
                           weight_decay = args['weight_decay'])
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
    #                                                      gamma = args['lr_decay'])

    # ----------- Make logdir -------------------------------------
    if not os.path.exists(os.path.join('logdir', args['save_dir'])):
        os.makedirs(os.path.join('logdir', args['save_dir']))
    summary = SummaryWriter(os.path.join('logdir', args['save_dir']))
    with open(os.path.join('logdir', args['save_dir'], 'model_config.json') ,'w') as f:
        json.dump(args, f, indent=4)
    
    train(args, model, optimizer, train_dloader, val_dloader, summary)
