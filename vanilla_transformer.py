import json, ast
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr

import config, dataset, model


def load_dataframes(max_samples=None, append_dev_set=True):
    tr_df = pd.read_csv(config.train_fname)
    valid_df = pd.read_csv(config.valid_fname)
    te_df = pd.read_csv(config.test_fname)

    if append_dev_set:
        tr_df = pd.concat([tr_df, valid_df])

    if max_samples:
        tr_df = tr_df.sample(n=max_samples)
        te_df = te_df.sample(n=max_samples)

    return tr_df, te_df


def get_metrics_from_logits(logits, labels, detach=True):
    if detach:
        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        corr, pval = spearmanr(logits,labels)
    else:
        corr, pval = spearmanr(logits, labels)

    return corr



def test(net, test_loader, device='cpu', pred_save_fname=False, return_preds= False):
    net.eval()
    with torch.no_grad():
        for batch_num, (seq, attn_masks, labels) in enumerate(tqdm(test_loader), start=1):
            seq, attn_masks = seq.cuda(device), attn_masks.cuda(device)
            logits = net(seq, attn_masks)
            preds = logits.detach().cpu().numpy().reshape(-1,)
            labels = labels.cpu().numpy()
            if batch_num == 1:
                all_trues = labels
                all_preds = preds
            else:
                all_trues = np.hstack([all_trues, labels])
                all_preds = np.hstack([all_preds, preds])

        corr = get_metrics_from_logits(all_preds, all_trues, detach=False)
        if pred_save_fname:
            with open(pred_save_fname,'w') as fh:
                for i in all_preds:
                    print(i,file=fh)
            print(f'Predictions are saved to file: {pred_save_fname}')

    if return_preds:
        return all_preds, corr
    else:
        return corr


def train_model(net, criterion, optimizer, scheduler, train_loader, test_loader=None,
                print_every=100, n_epochs=10, device='cpu', save_model=True,
                save_every=5):

    for e in range(1, n_epochs+1):
        t0 = time.perf_counter()
        e_loss = []
        for batch_num, (seq_attnmask_labels) in enumerate(tqdm(train_loader), start=1):
            # Clear gradients
            optimizer.zero_grad()

            #get the 3 input args for this batch
            seq, attn_mask, labels = seq_attnmask_labels

            # Converting these to cuda tensors
            seq, attn_mask, labels = seq.cuda(device), attn_mask.cuda(device), labels.cuda(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_mask)

            # Computing loss
            loss = criterion(logits.squeeze(1), labels)
            e_loss.append(loss.item())

            # Backpropagating the gradients for losses on all classes
            loss.backward()

            # Optimization step
            optimizer.step()
            scheduler.step()

            if batch_num % print_every == 0:
                corr = get_metrics_from_logits(logits, labels)
                print(f"batch {batch_num} of epoch {e} complete. Loss : {loss.item()} "
                      f"spearman corr: {corr:.4f}")

        t = time.perf_counter() - t0
        t = t/60 # mins
        e_loss = np.array(e_loss).mean()
        print(f'Done epoch: {e} in {t:.2f} mins, epoch loss: {e_loss}')

        if test_loader != None:
            if e%save_every == 0:
                pred_save_fname = config.PREDICTIONS_FNAME.replace('.txt', f'_{e}.txt')
            else:
                pred_save_fname = None
            corr = test(net, test_loader, device=device, pred_save_fname=pred_save_fname)
            print(f'After epoch: {e}, validation spearman corr: {corr:.4f}')

        if save_model:
            if e%save_every==0:
                save_fname = config.TRAINED_MODEL_FNAME_PREFIX + f'_e_{e}.pt'
                torch.save(net, save_fname)
                print(f'Saved model at: {save_fname} after epoch: {e}')



def main():
    max_samples = None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tr_df, val_df = load_dataframes(max_samples)
    print(f'Loaded training and validation DFs of shape: {tr_df.shape} and {val_df.shape}')

    #for dataset loader
    train_set = dataset.dataset(tr_df,
                                max_len=config.MAX_SEQ_LEN,
                                other_langs=True,
                                crosslingual=True)
    valid_set = dataset.dataset(val_df, max_len=config.MAX_SEQ_LEN, other_langs=False)

    #creating dataloader
    train_loader = DataLoader(train_set,
                              shuffle = True,#Training set should be shuffled
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_CPU_WORKERS)
    valid_loader = DataLoader(valid_set,
                             shuffle = False,#Validation set should NOT be shuffled
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_CPU_WORKERS)

    #creating BERT model
    if config.TRAINED_MODEL_FNAME:
        bert_model = torch.load(config.TRAINED_MODEL_FNAME)
        print(f'Loaded trained model: {bert_model} from file: {config.TRAINED_MODEL_FNAME}')
    else:
        bert_model = model.bert_classifier(freeze_bert=config.BERT_LAYER_FREEZE)
        print(f"created NEW TRANSFORMER model for finetuning: {bert_model}")
    bert_model.cuda()

    # Multi GPU setting
    if config.MULTIGPU:
        device_ids = [0, 1, 2, 3] #huggingface allows parallelizing only upto 4 cards
        bert_model = nn.DataParallel(bert_model, device_ids = device_ids)
        print(f'Model parallelized on the following cards: ',device_ids)

    #loss function (with weights)
    criterion = nn.MSELoss()

    param_optimizer = list(bert_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_set) / config.BATCH_SIZE * config.NUM_EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    train_model(bert_model, criterion, optimizer, scheduler, train_loader, valid_loader,
                print_every=config.PRINT_EVERY, n_epochs=config.NUM_EPOCHS, device=device,
                save_model=True, save_every = config.SAVE_EVERY)

    corr = test(bert_model, valid_loader, device=device, pred_save_fname=True)
    print(f'After ALL epochs: validation spearman corr: {corr:.4f}')


if __name__ == '__main__':
    main()