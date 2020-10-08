from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import ast, json, os, numpy as np


X_train_fname = 'data/train_pairs.txt'
X_valid_fname = 'data/valid_pairs.txt'
id_ques_map_fname = 'data/id_ques_map.json'
y_train_fname = 'data/y_train.txt'
y_valid_fname = 'data/y_valid.txt'

def load_id_ques_map(fname):
    with open(fname) as fh:
        id_ques_map = json.load(fh)
    id_ques_map = {int(k): str(v) for k, v in id_ques_map.items()}
    id_ques_map[0] = ''
    print(f'Loaded {max(id_ques_map.keys())} ids to questions from {fname}')
    return id_ques_map


def load_Xy(max_samples=None):
    train_pairs = np.array([list(ast.literal_eval(l.strip())) for l in
                            open(X_train_fname).readlines()])[:max_samples]
    valid_pairs = np.array([list(ast.literal_eval(l.strip())) for l in
                            open(X_valid_fname).readlines()])[:max_samples]

    id_ques_map = load_id_ques_map(id_ques_map_fname)
    X_train = [(id_ques_map[i], id_ques_map[j]) for i,j in train_pairs]
    X_valid = [(id_ques_map[i], id_ques_map[j]) for i,j in valid_pairs]

    y_train = np.loadtxt(y_train_fname).astype(int)[:max_samples]
    y_valid = np.loadtxt(y_valid_fname).astype(int)[:max_samples]

    return X_train, X_valid, y_train, y_valid

def main():
    max_samples = 1000
    use_model = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/4')

    X_train, X_valid, y_train, y_valid = load_Xy(max_samples)

    sents_0 = np.asarray([p[0] for p in X_train])
    sents_0 = use_model.encode(sents_0)
    print(sents_0.shape)
    sents_1 = np.asarray([p[1] for p in X_train])
    sents_1 = use_model.encode(sents_1)
    X_train = np.hstack((sents_0, sents_1))

    print(X_train)

if __name__ == '__main__':
    main()
