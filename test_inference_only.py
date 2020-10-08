import torch
import numpy as np, sys, time, ast
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd

import config
import dataset
from vanilla_transformer import test


def get_lang_report(df):
    res_dict = {}
    lang_pairs = [('ar', 'ar'), ('ar', 'en'), ('es', 'es'), ('es', 'en'), ('en', 'en'), ('tr', 'en')]
    res = []
    for p in lang_pairs:
        sdf = df[(df.l1 == p[0]) & (df.l2 == p[1])]
        scc = f'{sdf.score.corr(sdf["pred_score"], method="spearman")*100:.2f}'
        res.append(scc)
        res_dict[p] = scc

    overall_scc = f'{df.score.corr(df["pred_score"], method="spearman")*100:.2f}'
    res.append(overall_scc)
    res_dict['overall'] = overall_scc

    print(res_dict)
    report = f'{" ".join(res)}'
    return report



def main():
    #cmd line args
    test_fname = './data/2017/2017_multilingual_eval_set.csv'
    TRAINED_MODEL_FNAME = 'XLM-ROBERTA-BASE_model_e_2.pt' #sys.argv[3]
    TRAINED_MODEL_FNAME = 'PVL_LABSE_BERT_model_e_10.pt' #sys.argv[3]
    TRAINED_MODEL_FNAME = 'SENTENCE-TRANSFORMERS_XLM-R-100LANGS-BERT-BASE-NLI-STSB-MEAN-TOKENS_model_e_8.pt' #sys.argv[3]
    is_cpu = 0 #bool(int(sys.argv[4])) #0 or 1

    #Setting up the device to be used for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_cpu:
        device = torch.device('cpu')
    print(f'****Device used for this inference: {device}***')

    #Load the validation set
    test_df = pd.read_csv(test_fname)
    test_set = dataset.dataset(test_df, max_len=config.MAX_SEQ_LEN)

    # Validation set should NOT be shuffled
    test_loader = DataLoader(test_set,
                             shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #Loading BERT model
    bert_model = torch.load(TRAINED_MODEL_FNAME)
    try:
        bert_model = bert_model.module
        print('loaded model from bert_model.module')
    except:
        print('Unable to locate .module form the model loaded, '
              'this model might not be from nn.DataParallel')

    # Switching to GPU or CPU based on the cmd line arg
    bert_model.cuda()
    if is_cpu:
        bert_model.to('cpu')
    print(f'Loaded trained model: {bert_model} \nfrom file: {TRAINED_MODEL_FNAME}')


    # # Multi GPU setting - this is NOT relevant now
    # if config.MULTIGPU:
    #     device_ids = [0, 1, 2, 3] #huggingface allows parallelizing only upto 4 cards
    #     bert_model = nn.DataParallel(bert_model, device_ids = device_ids)
    #     print(f'Model parallelized on the following cards: ',device_ids)

    t0 = time.perf_counter()
    preds, corr = test(bert_model, test_loader, device=device, pred_save_fname=False, return_preds=True)
    test_df['pred_score'] = preds
    t = round(time.perf_counter() - t0, 2)
    print(f'Inference time: {t}, TEST spearman corr: {corr}')

    lang_report = get_lang_report(test_df)
    print(lang_report)


if __name__ == '__main__':
    main()