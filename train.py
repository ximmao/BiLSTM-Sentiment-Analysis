import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data, vocab
import spacy
from sklearn.metrics import accuracy_score

import os
import argparse

from BiLSTM_Senti import biLSTM_MLP

"""
dataset should be in ./dataset folder
model is saved in ./model folder
"""

def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

def OneHotToInt(batch_pred):
    """Revert one hot encodings numpy array to number numpy array"""
    pred_list = []
    for i in range(a.shape[0]):
        for j in range(a[i].shape[0]):
            if a[i,j] == 1:
               pred_list.append(j+1)
    return np.array(pred_list)

def calculate_accuracy(pred, real):
    #print(pred.size())
    pred_list = F.sigmoid(pred).cpu().data.numpy().reshape(-1)
    int_list = (pred_list > 0.5).astype(int).tolist()
    return accuracy_score(np.array(int_list), real, normalize=False)


parser = argparse.ArgumentParser()

parser.add_argument('--datasetToTrain', type=str, default='IMDB', choices=['IMDB', 'Twitter'])
parser.add_argument('--levelOnHypernym', type=str, default='NoneUsed', choices=['NoneUsed', 'ToRoot', 'KeepBoth'])

config = parser.parse_args()
print(config)

tokenizer_en = spacy.load('en')
def tokenizer(text):
    return [w.text.lower() for w in tokenizer_en(text.strip())]

print("start")

text_field = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True)
label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

data_val_fields = [('text', text_field), ('label', label_field)]

DATASET = config.datasetToTrain
HYPERNYM = config.levelOnHypernym
print("Making Datasets")
print("Training Dataset is " + DATASET)
print("Hypernym Level is " + HYPERNYM)

if HYPERNYM == 'NoneUsed':
    PREFIX = DATASET
elif HYPERNYM == "ToRoot":
    PREFIX = DATASET + '-root'
elif HYPERNYM == "KeepBoth":
    PREFIX = DATASET + '-both'

train_dataset, valid_dataset = data.TabularDataset.splits(path='./dataset',
                                                          format='csv',
                                                          train=PREFIX+'-train.csv',
                                                          validation=PREFIX+'-valid.csv',
                                                          fields=data_val_fields,
                                                          skip_header=True)

text_field.build_vocab(train_dataset, max_size=20000, vectors="glove.6B.100d")
label_field.build_vocab(train_dataset)
print("text vocabulary length", len(text_field.vocab))
print("label vocabulary length", len(label_field.vocab), label_field.vocab)

if HYPERNYM == 'NoneUsed':
    AFFIX = ''
elif HYPERNYM == "ToRoot":
    AFFIX = '-root'
elif HYPERNYM == "KeepBoth":
    AFFIX = '-both'

test_dataset_twitter, test_dataset_imdb = data.TabularDataset.splits(path='./dataset',
                                                                     format='csv',
                                                                     validation='Twitter'+AFFIX+'-test.csv',
                                                                     test='IMDB'+AFFIX+'-test.csv',
                                                                     fields=data_val_fields,
                                                                     skip_header=True)

print("text", train_dataset[0].text)
print("label", train_dataset[0].label)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EMBEDDING = 100
HIDDEN_DIM = 256
USE_BIDIRECTIONAL = True

train_iterator, valid_iterator, test_iterator_imdb, test_iterator_twitter = data.BucketIterator.splits(
    datasets=(train_dataset, valid_dataset, test_dataset_imdb, test_dataset_twitter),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    device=device,
    sort_within_batch=True,
    repeat=False)


model = biLSTM_MLP(EMBEDDING, BATCH_SIZE, len(text_field.vocab), HIDDEN_DIM, 1, USE_BIDIRECTIONAL)
print_network(model)
pretrained_embeddings = text_field.vocab.vectors
model.WordEmbedding_default.weight.data.copy_(pretrained_embeddings)
print(pretrained_embeddings.shape)
if torch.cuda.is_available():
    model = model.cuda()
optim_bilstm = optim.Adam(model.parameters(), lr=1e-4)
loss = nn.BCEWithLogitsLoss(size_average=True)

model_save_dir = "./model"
Test_index = 1
epoch = 0
valid_loss_last5 = []
valid_stop_check = []
BreakLoop = False
for epoch in range(1, 50):
    model.train()
    train_accu_cnt = 0
    minibatch_losses_t = []
    for batch in train_iterator:
        optim_bilstm.zero_grad()
        pred = model(batch.text)
        senti_temp = batch.label.unsqueeze(1).float()
        loss_t = loss(pred, senti_temp)
        loss_t.backward()
        minibatch_losses_t.append(loss_t.item())
        optim_bilstm.step()
        accu = calculate_accuracy(pred, senti_temp.cpu().data.numpy())
        train_accu_cnt += accu

    model.eval()
    valid_accu_cnt = 0
    minibatch_losses_v = []
    with torch.no_grad():
        for batch in valid_iterator:
            pred = model(batch.text)
            senti_temp = batch.label.unsqueeze(1).float()
            loss_v = loss(pred, senti_temp)
            accu = calculate_accuracy(pred, senti_temp.cpu().data.numpy())
            minibatch_losses_v.append(loss_v.item())
            valid_accu_cnt += accu
    print("epoch", epoch, "train losses", np.mean(np.array(minibatch_losses_t)), "train accuracy", train_accu_cnt/len(train_dataset), "valid losses", np.mean(np.array(minibatch_losses_v)), "valid accuracy", valid_accu_cnt/len(valid_dataset))
    valid_loss_last5.append(np.mean(np.array(minibatch_losses_v)))

    if len(valid_loss_last5) > 5:
        valid_loss_last5 = valid_loss_last5[-5:]
    if len(valid_loss_last5) == 5:
        valid_stop_check.append(np.mean(np.array(valid_loss_last5)))
        if len(valid_stop_check) > 4:
            valid_stop_check = valid_stop_check[-4:]
            for i in range(3):
                if valid_stop_check[i] < valid_stop_check[i+1]:
                    BreakLoop = True
                else:
                    BreakLoop = False
    if BreakLoop == True:
        model_path = os.path.join(model_save_dir, '{}-{}-BiLSTM.ckpt'.format(DATASET, epoch))
        torch.save(model.state_dict(), model_path)
        print('Saved model into', model_path)
        break

test_accu_cnt_i = 0
test_accu_cnt_a = 0
minibatch_losses_i = []
minibatch_losses_a = []
with torch.no_grad():
    for batch in test_iterator_imdb:
        pred = model(batch.text)
        senti_temp = batch.label.unsqueeze(1).float()
        loss_i = loss(pred, senti_temp)
        accu = calculate_accuracy(pred, senti_temp.cpu().data.numpy())
        minibatch_losses_i.append(loss_i.item())
        test_accu_cnt_i += accu
    print("test losses IMDB", np.mean(np.array(minibatch_losses_i)), "test accuracy IMDB", test_accu_cnt_i/len(test_dataset_imdb))
    for batch in test_iterator_twitter:
        pred = model(batch.text)
        senti_temp = batch.label.unsqueeze(1).float()
        loss_a = loss(pred, senti_temp)
        accu = calculate_accuracy(pred, senti_temp.cpu().data.numpy())
        minibatch_losses_a.append(loss_a.item())
        test_accu_cnt_a += accu
    print("test losses Twitter", np.mean(np.array(minibatch_losses_a)), "test accuracy Twitter", test_accu_cnt_a/len(test_dataset_twitter))
