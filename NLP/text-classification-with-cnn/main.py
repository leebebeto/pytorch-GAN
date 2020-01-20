import torch
import random
from random import *
from collections import Counter
import argparse
import numpy as np
import csv
import re
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from random import shuffle
from collections import OrderedDict 
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_Classifier(nn.Module):

	def __init__(self):
		super(CNN_Classifier, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.layer_3 = nn.Conv2d(1, 100, kernel_size = (3, embedding_dim))
		self.layer_4 = nn.Conv2d(1, 100, kernel_size = (4, embedding_dim))
		self.layer_5 = nn.Conv2d(1, 100, kernel_size = (5, embedding_dim))
		self.relu = nn.ReLU(inplace = True)
		self.dropout = nn.Dropout(p = 0.5) 
		self.softmax = nn.Softmax(dim = 1)
		self.W_out = nn.Linear(embedding_dim,num_classes)

	def forward(self, lookup_tensor):
		embedded_tensor = self.embedding(lookup_tensor).cpu()
		x = Variable(embedded_tensor, requires_grad =True).cpu()
		x = x.unsqueeze(1)
		out_3 = self.layer_3(x)
		out_3 = self.relu(out_3).squeeze(3)
		out_4 = self.layer_4(x)
		out_4 = self.relu(out_4).squeeze(3)
		out_5 = self.layer_5(x)
		out_5 = self.relu(out_5).squeeze(3)
		out_3 = F.max_pool1d(out_3, out_3.size(2)).squeeze(2)
		out_4 = F.max_pool1d(out_4, out_4.size(2)).squeeze(2)
		out_5 = F.max_pool1d(out_5, out_5.size(2)).squeeze(2)
		out_3 = self.dropout(out_3)
		out_4 = self.dropout(out_4)
		out_5 = self.dropout(out_5)
		flatten = torch.cat([out_3, out_4, out_5], 1).cpu()
		output = self.W_out(flatten)
		output = self.softmax(output)
		output = output.squeeze(0)
		return output

mode = 'CNN-rand'
num_classes = 2
dropout = 0.5
batch_size = 50
embedding_dim = 300
learning_rate = 0.001
num_epochs = 10
criterion = nn.CrossEntropyLoss()

positive_data = open('rt-polarity.pos.txt',mode='r').readlines()
negative_data = open('rt-polarity.neg.txt',mode='r').readlines()


corpus, sentences, labels = [], [], []
for data in [positive_data, negative_data]:
	for line in data:
		if len(line) > 5:
			sentences.append(line)
		if data == positive_data:
			labels.append(0)
		elif data == negative_data:
			labels.append(1)
		else:
			print('error')
		corpus += line.split(' ') 

ind2sentence_temp = {i: sentence for i, sentence in enumerate(sentences)}
ind2label_temp = {i: label for i, label in enumerate(labels)}

temp_sentence = list(ind2sentence_temp.items())

ind2sentence = OrderedDict(temp_sentence)
ind2label = {}
for k in ind2sentence.keys():
	ind2label[k] = ind2label_temp[k]

assert len(ind2sentence) == len(ind2label)

total_step = int(len(ind2sentence)/batch_size)

frequency = Counter(corpus)
processed_dict = {}
processed = []
for k,v in frequency.items():
	if v > 4:
		processed_dict[k] = v
		processed.append(k)
vocabulary = list(set(processed))
vocab_size = len(corpus)
ind2word, word2ind = {}, {}
for i, word in enumerate(vocabulary):
	ind2word[i] = word
	word2ind[word] = i
model = CNN_Classifier().to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
total_index_list = list(ind2sentence.keys())
shuffle(total_index_list)
train_index_list = total_index_list[:int(len(total_index_list)*0.8)]

test_index_list = total_index_list[int(len(total_index_list)*0.8)+1:]

for epoch in range(num_epochs):
	for i in range(0,len(train_index_list), batch_size):
		input_data = []
		target_label = []
		for iteration in range(i, i+batch_size):
			try:
				iteration = train_index_list[iteration]
			except:
				continue

			sentence = ind2sentence[iteration].split(' ')
			label = ind2label[iteration]
			target_label.append(label)
			word_list = []
			for word in sentence:
				try:
					word_list.append(word2ind[word])
				except:
					pass
			input_data.append(word_list)
		max_list = max(input_data, key = lambda i: len(i)) 

		for input_data_list in input_data:
			while len(input_data_list) < len(max_list):
				input_data_list.append(0)
		input_data = torch.tensor(input_data).to(device)
		target_label = torch.LongTensor(target_label).to(device)

		outputs = model(input_data).to(device)
		losses = criterion(outputs, target_label)

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()


		if i % 500 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i, len(train_index_list), losses.item()))

with torch.no_grad():
	model.eval()
	correct = 0
	total = len(test_index_list)
	for i in range(len(test_index_list)):
		input_data = []
		target_label = []
		sentence = ind2sentence[i].split(' ')
		label = ind2label[i]
		target_label.append(label)
		word_list = []
		for word in sentence:
			try:
				word_list.append(word2ind[word])
			except:
				pass
		if len(word_list) <= 5:
			continue

		input_data.append(word_list)
		max_list = max(input_data, key = lambda i: len(i)) 

		for input_data_list in input_data:
			while len(input_data_list) < len(max_list):
				input_data_list.append(0)
		input_data = torch.tensor(input_data).to(device)
		target_label = torch.LongTensor(target_label).to(device)
		outputs = model(input_data).to(device)

		_, predicted = torch.max(outputs.data, 0)
		correct += (label == predicted).item()
		if i%500 == 0:
			print('{}th testing, correct : {}'.format(i,correct))

	print('Accuracy of the network on the {} test images: {} %'.format(total ,100 * correct / total))

"Accuracy recorded 86.491%"
