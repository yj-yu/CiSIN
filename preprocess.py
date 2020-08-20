'''
old
\# of total_data:  5757 yes, no verb 19247 47 yes, no gen 18871 376
100%|█████████████████████████| 8/8 [00:03<00:00,  2.03it/s]
\# of total_data:  547 yes, no verb 1772 8 yes, no gen 1770 2
'''

import pickle
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from glob import glob
import os
import json

base_dir = '/home_kahlo/jongseok.kim/dataset/LSMDC/' 

gender_dir = base_dir + 'supplementary/csv_ids'
mvad_pkl = {'train':base_dir + 'features_mvad/MVAD_train_hbid.pkl',
            'val':base_dir + 'features_mvad/MVAD_val_hbid.pkl',
            'test':base_dir + 'features_mvad/MVAD_test_hbid.pkl'}
dep_pkl = base_dir + 'features_mvad/MVAD_dep.pkl'

def preprocess(mode = 'train'):
    # --- Load MPII data --------------------------
    with open(mvad_pkl[mode], 'rb') as f:
        mvad_data = pickle.load(f)
    with open(dep_pkl, 'rb') as f:
        dep_data = pickle.load(f)

    # start
    datas, fails, yes_verb, no_verb, yes_gen, no_gen = [], 0, 0, 0, 0, 0
    hN = 8
    for mov_id in tqdm(mvad_data, ncols=60):
        # Character Gender
        lsmdc_gender_csv = glob(os.path.join(gender_dir, '*'+mov_id+'-characters.csv'))
        if len(lsmdc_gender_csv) == 0:
            name2per = {}
        else:
            with open(lsmdc_gender_csv[0], encoding='utf-8') as f:
                gender_csv = f.readlines()
            per2gen = {}
            for gc in gender_csv:
                _, _, gender, person = gc.strip().split('\t')
                per2gen[person] = 0 if gender == 'M' else 1 # M: 0, F: 1, UNk: 2
            with open(os.path.join(base_dir + 'supplementary/mvad_to_lsmdc', mov_id+'.json')) as f:
                name2per = json.load(f)
            for name_key in name2per:
                name2per[name_key] = per2gen[name2per[name_key]]

        # Start
        agg5, sN = [], 0
        clip_ids = list(mvad_data[mov_id].keys())
        clip_ids.sort(key = lambda x: (len(x), x))

        for clip_id in clip_ids:
            sent = mvad_data[mov_id][clip_id]['sentence']
            heads = mvad_data[mov_id][clip_id]['heads']
            # --- Tokenize sentence -----------------------
            word_list, someone_info = [], []
            sent_splits = sent.split(' ')

            for si, word in enumerate(sent_splits):
                if '<PERSON>' in word:
                    if '</PERSON>' not in word:
                        word += ' ' + sent_splits[si + 1]
                        sent_splits[si + 1] = ''
                    # e.g. <PERSON>RICHARDSON</PERSON>
                    names = word.replace('<PERSON>','').split('</PERSON>')
                    for head_id in heads: ## There are some id issiue but ignore it..
                        if names[0].lower() in head_id.lower():
                            #import pudb;pudb.set_trace()
                            hb_id =  max(heads[head_id], key= lambda x: x[1])[0] # Select with longest trajs
                            nl = names[0].lower()
                            if nl in dep_data[clip_id]['roots']:
                                someone_info.append({
                                    'name': head_id,
                                    'hb_id': hb_id,
                                    'loc': len(word_list),
                                    'gender': name2per.get(head_id, 2),
                                    'dep': dep_data[clip_id]['roots'][nl]['parent']['idx']
                                })
                                yes_verb += 1
                            else:
                                no_verb += 1

                            if head_id in name2per:
                                yes_gen += 1
                            else:
                                no_gen += 1
                            break
                    word = 'someone' + names[1]
                if len(word) > 0:
                    word_list.extend(word_tokenize(word))

            if len(someone_info) > hN:
                someone_info = someone_info[:hN]
            sN += len(someone_info)
            agg5.append({'mov_id': mov_id,
                         'clip_id': clip_id,
                         'word_list': word_list,
                         'someone_info':someone_info, 
                         'sent': sent})
            # If 5, aggregate and append into datas
            if len(agg5) == 5:
                if sN > 0:
                    datas.append(agg5)
                else:
                    fails += 1
                agg5, sN = [], 0
    print("\# of total_data: ", len(datas),"yes, no verb", yes_verb, no_verb, "yes, no gen", yes_gen, no_gen)
    return datas

for mode in ['train', 'val']:
    datas = preprocess(mode)
    with open('/home_kahlo/jongseok.kim/dataset/LSMDC/features_mvad/MVAD_%s_agg.pkl'%mode, 'wb') as f:
        pickle.dump(datas, f)
'''
[[
    {mov_id: str
     clip_id: str,
     word_list: [str],
     someone_info: [
         {name: str,
          hb_id: int,
          loc: int,
          gender: int (0:M, 1:F, 2:no data),
          dep: int,
         }, ...
     ]
     sent: str (someone as [SOMEONE]),
     
    }, ...
],]
'''


