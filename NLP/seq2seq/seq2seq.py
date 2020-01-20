from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import Counter
import argparse
import numpy as np
import csv
import pandas as pd
from torch.autograd import Variable
from collections import OrderedDict 
import os
import copy

# loss 함수: softmax -> logsoftmax + cell을 decoder_hidden = decoder_hidden 일케 업데이트를 안해줬었음

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description = "Seq2seq translation based on attention")
parser.add_argument('--batch_size', type = int, default = 1, help = 'batch_size')
parser.add_argument('--input_size', type = int, default = 256, help = 'input_size')
parser.add_argument('--embedding_size', type = int, default = 256, help = 'embedding size of your encoder  and decoder')
parser.add_argument('--hidden_size', type = int, default = 128, help = 'hidden_size of your LSTM cell')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning_rate')
parser.add_argument('--beta', type = float, default = (0.9, 0.99), help = 'decay_rate')
parser.add_argument('--epoch', type = int, default = 5, help = 'epoch')

args = parser.parse_args()


# pre-processing codes from Pytorch Seq2seq Tutorial "https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        self.word2index["SOS"] = 0
        self.word2index["EOS"] = 1
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# Tutorial codes end here. 


class Encoder(nn.Module):
    def __init__(self, input_lang):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_lang.n_words, args.embedding_size)
        self.lstm = nn.LSTM(input_size = args.input_size, num_layers=1, hidden_size = args.hidden_size, batch_first = True)

    def forward(self, data, h_n, c_n):
        data = self.embedding(data).unsqueeze(0)
        output, (h_n, c_n) = self.lstm(data, (h_n, c_n))
        return output, (h_n, c_n)


class Decoder(nn.Module):
    def __init__(self, output_lang):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_lang.n_words, args.embedding_size)
        self.lstm = nn.LSTM(input_size = args.input_size, num_layers=1, hidden_size = args.hidden_size, batch_first = True)
        self.fc = nn.Linear(args.hidden_size, output_lang.n_words)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, data, hidden_state):
        data = self.embedding(data).unsqueeze(0)
        h_n, c_n = hidden_state
        output, (h_n, c_n) = self.lstm(data, (h_n, c_n))
        output = self.fc(self.relu(output))
        output = output.squeeze(0)
        output = self.softmax(output)
        return output, (h_n, c_n)

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)


def sentence2index_list(input_sentence, output_sentence):
    encoder_list, decoder_list = [], []
    for i, sentence in enumerate([input_sentence, output_sentence]):
        for word in sentence.split(' '):
            if i == 0:
                encoder_list.append(input_lang.word2index[word])
            else:
                decoder_list.append(output_lang.word2index[word])
    return encoder_list, decoder_list

encoder = Encoder(input_lang)
decoder = Decoder(output_lang)
h_n = torch.zeros(1,args.batch_size, args.hidden_size)
c_n = torch.zeros(1,args.batch_size, args.hidden_size)

criterion = nn.CrossEntropyLoss()
encoder_optimizer = torch.optim.Adam(params = encoder.parameters(), lr = args.learning_rate, betas = args.beta)
decoder_optimizer = torch.optim.Adam(params = decoder.parameters(), lr = args.learning_rate, betas = args.beta)

for i in range(args.epoch):
    for sentences in pairs:
        result_index = []
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0
        encoder_list, decoder_list = sentence2index_list(sentences[0], sentences[1])
        decoder_input_list, decoder_target_list = copy.deepcopy(decoder_list), copy.deepcopy(decoder_list) 
        decoder_input_list.insert(0, output_lang.word2index['SOS'])
        decoder_target_list.append(output_lang.word2index['EOS'])
        for word in encoder_list:
            encoder_input_tensor =torch.tensor([word])
            output, (h_n, c_n) = encoder(encoder_input_tensor, h_n, c_n)

        decoder_hidden_state = (h_n, c_n)
        decoder_input_tensor = torch.tensor([decoder_list[0]])
        for index in range(len(decoder_list)):
            decoder_target_tensor = Variable(torch.tensor([decoder_target_list[index]]))
            output, decoder_hidden_state = decoder(decoder_input_tensor, decoder_hidden_state)
            decoder_input_tensor = torch.tensor([output.topk(1)[1].squeeze().detach()])
            # print('decoder_target_tensor', decoder_target_tensor)
            # print(output.topk(1)[1])
            result_index.append(output.topk(1)[1].squeeze(0).item())
            loss += criterion(output, decoder_target_tensor)

        result = [output_lang.index2word[i] for i in result_index]
        print(sentences[1])
        print('result', result)
        loss.backward(retain_graph = True)
        encoder_optimizer.step()
        decoder_optimizer.step()


        print(loss / len(decoder_list))













































