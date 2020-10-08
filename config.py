import psutil
import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MODEL_NAME =  'pvl/labse_bert'#, 'bert-base-uncased',  'albert-base-v2'
# MODEL_NAME =  'xlm-roberta-base'
MODEL_NAME =  "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
BATCH_SIZE = 30
LR = 0.00001

data_folder = './data/stsbenchmark'
train_fname = os.path.join(data_folder, 'pp_stsb_train_w_trans.csv')
valid_fname = os.path.join(data_folder, 'pp_stsb_dev_w_trans.csv')
test_fname = os.path.join(data_folder, 'pp_stsb_test_w_trans.csv')

# IS_LOWER = True if 'uncased' in MODEL_NAME else False
IS_LOWER = True #for LaBSE

MAX_SEQ_LEN = 256 #for LaBSE
NUM_EPOCHS = 20
NUM_CPU_WORKERS = psutil.cpu_count()
PRINT_EVERY = 100
SAVE_EVERY = 1

BERT_LAYER_FREEZE = False

#when using xlarge vs 16x large AWS m/c
MULTIGPU = True if torch.cuda.device_count() > 1 else False

TRAINED_MODEL_FNAME_PREFIX = MODEL_NAME.replace('/','_').upper()+'_model'
TRAINED_MODEL_FNAME = None #'PVL_LABSE_BERT_model_e_4.pt'

CONTEXT_VECTOR_SIZE = 1024 if 'large' in MODEL_NAME else 768
PREDICTIONS_FNAME = '_'.join([MODEL_NAME.replace('/','_').upper(),'preds.txt'])





