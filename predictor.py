import os
import re
import json
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
import csv
import itertools
import nltk
from collections import defaultdict, OrderedDict
import flask
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import scispacy
import spacy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
## Setup Tokenizers
"""

# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

"""
## Define model
"""


import os
import re
import csv
import itertools

import nltk
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_LEN = 512
BATCH_SIZE = 32
tokenizer = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

tag_values=['I-Cellular_component',
 'E-Gene_or_gene_product',
 'I-Organism_subdivision',
 'I-Organism_substance',
 'B-Gene_or_gene_product',
 'B-Cancer',
 'I-Cancer',
 'E-Pathological_formation',
 'I-Pathological_formation',
 'S-Organism_substance',
 'S-Organ',
 'E-Organ',
 'I-Immaterial_anatomical_entity',
 'E-Cell',
 'I-Simple_chemical',
 'E-Tissue',
 'B-Organism',
 'S-Cellular_component',
 'S-Pathological_formation',
 'I-Amino_acid',
 'E-Anatomical_system',
 'S-Developing_anatomical_structure',
 'B-Immaterial_anatomical_entity',
 'B-Protein',
 'I-Chemical',
 'S-Organism',
 'I-Gene_or_gene_product',
 'I-Cell',
 'E-Multi-tissue_structure',
 'B-Organism_subdivision',
 'E-Cellular_component',
 'S-Chemical',
 'S-Protein',
 'B-Simple_chemical',
 'E-Organism',
 'B-Developing_anatomical_structure',
 'S-Multi-tissue_structure',
 'S-Immaterial_anatomical_entity',
 'B-Organism_substance',
 'E-Organism_substance',
 'E-Simple_chemical',
 'I-Tissue',
 'E-Immaterial_anatomical_entity',
 'I-Organism',
 'I-Protein',
 'S-Organism_subdivision',
 'E-Cancer',
 'I-Developing_anatomical_structure',
 'S-Tissue',
 'E-Chemical',
 'S-Amino_acid',
 'O',
 'S-Gene_or_gene_product',
 'E-Organism_subdivision',
 'B-Anatomical_system',
 'B-Chemical',
 'B-Cell',
 'E-Developing_anatomical_structure',
 'I-Multi-tissue_structure',
 'B-Pathological_formation',
 'B-Cellular_component',
 'B-Organ',
 'I-Anatomical_system',
 'S-Cell',
 'E-Amino_acid',
 'B-Tissue',
 'S-Simple_chemical',
 'E-Protein',
 'B-Multi-tissue_structure',
 'I-Organ',
 'S-Cancer',
 'B-Amino_acid',
 'S-Anatomical_system',
 'PAD']
vocab_len = len(tag_values)
df = pd.DataFrame({'tags':tag_values})


class SentenceFetch(object):
  
  def __init__(self, data):
    self.data = data
    self.sentences = []
    self.tags = []
    self.sent = []
    self.tag = []
    
    # make tsv file readable
    with open(self.data) as tsv_f:
      reader = csv.reader(tsv_f, delimiter='\t')
      for row in reader:
        if len(row) == 0:
          if len(self.sent) != len(self.tag):
            break
          self.sentences.append(self.sent)
          self.tags.append(self.tag)
          self.sent = []
          self.tag = []
        else:
          self.sent.append(row[0])
          self.tag.append(row[1])   

  def getSentences(self):
    return self.sentences
  
  def getTags(self):
    return self.tags


corpora = 'BioNLP'
sentences = []
tags = []
for subdir, dirs, files in os.walk(corpora):
    for file in files:
        if file == 'train.tsv':
            path = os.path.join(subdir, file)
            sent = SentenceFetch(path).getSentences()
            tag = SentenceFetch(path).getTags()
            sentences.extend(sent)
            tags.extend(tag)
            
sentences = sentences[0:20000]
tags = tags[0:20000]

def tok_with_labels(sent, text_labels):
  '''tokenize and keep labels intact'''
  tok_sent = []
  labels = []
  for word, label in zip(sent, text_labels):
    tok_word = tokenizer.tokenize(word)
    n_subwords = len(tok_word)

    tok_sent.extend(tok_word)
    labels.extend([label] * n_subwords)
  return tok_sent, labels

tok_texts_and_labels = [tok_with_labels(sent, labs) for sent, labs in zip(sentences, tags)]

tok_texts = [tok_label_pair[0] for tok_label_pair in tok_texts_and_labels]
labels = [tok_label_pair[1] for tok_label_pair in tok_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tag_values = list(set(itertools.chain.from_iterable(tags)))
tag_values.append("PAD")

tag2idx = {t: i for i,t in enumerate(tag_values)}

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

config = BertConfig.from_json_file('biobert_v1.1_pubmed/config.json')
tmp_d = torch.load('biobert_v1.1_pubmed/pytorch_model.bin', map_location=device)
state_dict = OrderedDict()

for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]



class BioBertNER(nn.Module):

  def __init__(self, vocab_len, config, state_dict):
    super().__init__()
    self.bert = BertModel(config)
    self.bert.load_state_dict(state_dict, strict=False)
    self.dropout = nn.Dropout(p=0.3)
    self.output = nn.Linear(self.bert.config.hidden_size, vocab_len)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
    encoded_layer, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    encl = encoded_layer[-1]
    out = self.dropout(encl)
    out = self.output(out)
    return out, out.argmax(-1)

model = BioBertNER(vocab_len,config,state_dict)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
epochs = 3
max_grad_norm = 1.0

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    for step,batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs,y_hat = model(b_input_ids,b_input_mask)
        
        _,preds = torch.max(outputs,dim=2)
        outputs = outputs.view(-1,outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1)
        loss = loss_fn(outputs,b_labels_shaped)
        correct_predictions += torch.sum(preds == b_labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double()/len(data_loader) , np.mean(losses)


def model_eval(model,data_loader,loss_fn,device):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
        
            outputs,y_hat = model(b_input_ids,b_input_mask)
        
            _,preds = torch.max(outputs,dim=2)
            outputs = outputs.view(-1,outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1)
            loss = loss_fn(outputs,b_labels_shaped)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())
        
    
    return correct_predictions.double()/len(data_loader) , np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
normalizer = BATCH_SIZE*MAX_LEN
loss_values = []

for epoch in range(epochs):
    
    total_loss = 0
    print(f'======== Epoch {epoch+1}/{epochs} ========')
    train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,device,scheduler)
    train_acc = train_acc/normalizer
    print(f'Train Loss: {train_loss} Train Accuracy: {train_acc}')
    total_loss += train_loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)  
    loss_values.append(avg_train_loss)
    
    val_acc,val_loss = model_eval(model,valid_dataloader,loss_fn,device)
    val_acc = val_acc/normalizer
    print(f'Val Loss: {val_loss} Val Accuracy: {val_acc}')
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

import nltk
nltk.download('punkt')

def tokenize_and_preserve(sentence):
    tokenized_sentence = []

    for word in sentence:
        tokenized_word = tokenizer.tokenize(word)   
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence

def bio_bert_ner_model(text):
    sent_text = nltk.sent_tokenize(text)
    tokenized_text = []
    for sentence in sent_text:
        tokenized_text.append(nltk.word_tokenize(sentence))

    tok_texts = [
        tokenize_and_preserve(sent) for sent in tokenized_text
    ]
    input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts]
    input_attentions = [[1]*len(in_id) for in_id in input_ids]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[1])
    new_tokens, new_labels = [], []
    for token in tokens:
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:

            new_tokens.append(token)
    actual_sentences = []
    pred_labels = []
    for x,y in zip(input_ids,input_attentions):
        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        x = x.view(-1,x.size()[-1])
        y = y.view(-1,y.size()[-1])
        with torch.no_grad():
            _,y_hat = model(x,y)
        label_indices = y_hat.to('cpu').numpy()
        tokens = tokenizer.convert_ids_to_tokens(x.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
        actual_sentences.append(new_tokens)
        pred_labels.append(new_labels)
    bio_ner_output=[]
    for token, label in zip(actual_sentences, pred_labels):
        for t,l in zip(token,label):
            if l != "O":
               bio_ner_output.append([t,l])
    return bio_ner_output



#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')




# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        spacy_english
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    data = flask.request.get_json()

    model_name_list=data['modelnames']
    original_sentence=data['sentence']
    output=[]
    for model_name in model_name_list:    



        if model_name=='bio_bert':
            ners=bio_bert_ner_model(original_sentence)
            output.append({"model_name":ners})

    final_output=json.dumps(output)
    return flask.Response(response=final_output, status=200, mimetype='application/json')

    