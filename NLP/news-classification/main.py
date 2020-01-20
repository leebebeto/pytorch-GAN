import torch
import random
from random import *
from collections import Counter
import argparse
import numpy as np
import statistics
from statistics import mode 
import csv
import re
import pandas as pd

def word2vec_trainer(processed, corpus_list, ind2gram, gram2ind, mode="CBOW", dimension=100, learning_rate=0.1):

# Xavier initialization of weight matrices
	W_emb = torch.randn(len(ind2gram), dimension) / (dimension**0.5)
	W_out = torch.randn(dimension, 4) / (dimension**0.5)
	print(W_emb.shape)
	print(W_out.shape)
	losses=[]
	for epoch in range(5):
		print("length of corpus list", len(corpus_list))
		for sentence_ind in range(len(corpus_list)):
			inputWordVector = list(np.zeros((1, W_emb.shape[0])))
			outputWordVector = list(np.zeros((1, W_out.shape[1])))

			gram_list = []
			title = corpus_list[sentence_ind]['title']
			title = title.lower()
			description = corpus_list[sentence_ind]['description']
			description = description.lower()
			input_gram_list = gram_words(title, -1, gram_mode) + gram_words(description, -1, gram_mode)
			input_gram_list = [str(word).split(',')[0][1:] + str(word).split(',')[1][:-1] for word in input_gram_list]
			input_ind = [gram2ind[gram] for gram in input_gram_list]	
			label= int(corpus_list[sentence_ind]['label'])-1
			for ind in input_ind:
				inputWordVector[0][ind] = 1	
			outputWordVector[0][label] = 1
			
			inputWordVector = torch.FloatTensor(inputWordVector[0])
			outputWordVector = torch.FloatTensor(outputWordVector[0])
			hidden_layer = torch.matmul(torch.t(W_emb),inputWordVector)
			output_layer = torch.matmul(torch.t(W_out),hidden_layer)    
			e = torch.exp(output_layer)
			final_layer = e / torch.sum(e)
			loss = torch.mean(-torch.log(final_layer[label]+1e-7))

			dfinal_layer = final_layer
			dfinal_layer[label] -= 1
			grad_out = torch.from_numpy(np.outer(hidden_layer.numpy(), dfinal_layer.numpy()))    
			grad_emb = torch.from_numpy(np.outer(inputWordVector.numpy(), np.dot(W_out,dfinal_layer.numpy().T)))
			W_emb -= learning_rate*grad_emb
			W_out -= learning_rate*grad_out

			losses.append(loss.item())
			if sentence_ind % 500==0:
				avg_loss=sum(losses)/len(losses)
				print("Epoch: %d || Iteration: %d || Loss : %f" %(epoch, sentence_ind, avg_loss,))
				losses=[]
	return W_emb, W_out

def sim(test_file, gram2ind, W_emb, W_out):
	row_count = test_file.shape[0]
	acc = 0
	for test_ind in range(row_count):
		# test_ind = randint(0, row_count-1)
		test_title =  test_file.iloc[test_ind][1]
		test_description = test_file.iloc[test_ind][2]
		test_title_gram = gram_words(test_title,-1, gram_mode)
		test_description_gram = gram_words(test_description,-1, gram_mode)
		test_input_gram = test_title_gram + test_description_gram

		inputWordVector = list(np.zeros((1, W_emb.shape[0])))
		outputWordVector = list(np.zeros((1, W_out.shape[1])))
		test_input_gram = [str(word).split(',')[0][1:] + str(word).split(',')[1][:-1] for word in test_input_gram]
		test_input_ind = []
		for gram in test_input_gram:
			try:
				test_input_ind.append(gram2ind[gram])
			except:
				pass

		test_label =  test_file.iloc[test_ind][0]
		for ind in test_input_ind:
			inputWordVector[0][ind] = 1	

		inputWordVector = torch.FloatTensor(inputWordVector[0])
		outputWordVector = torch.FloatTensor(outputWordVector[0])
		hidden_layer = torch.matmul(torch.t(W_emb),inputWordVector)
		output_layer = torch.matmul(torch.t(W_out),hidden_layer)    
		e = torch.exp(output_layer)
		final_layer = e / torch.sum(e)

		_ , pred_label = torch.max(final_layer, 0)
		pred_label = pred_label.item()
		test_label = str(test_label)
		pred_label = str(pred_label+1)
		if test_label == pred_label:
			acc +=1
		else:
			acc +=0

	acc = (acc / row_count ) * 100
	print('accuracy of this fast text classification is %.2f percent' % acc)





def gram_words(sentence, iteration, n):
	p = re.compile('[a-z]+')
	sentence = sentence.lower()
	sentence = p.findall(sentence)
	if iteration != -1:
		return {str(sentence[i:i+n]) : iteration for i in range(len(sentence)-1)}
	else:
		return [sentence[i:i+n] for i in range(len(sentence)-1)]

	

def main():

	print("loading...")
	text_dict = {}
	total_corpus = []
	text_set_1, text_set_2, text_set_3, text_set_4 = [], [], [], []
	text_dict = {}
	global gram_mode
	gram_mode = 2

	with open('ag_news_csv/train.csv', newline='') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for i, row in enumerate (csv_reader):
			temp_dict = {}
			temp_dict['label'] = row[0]
			temp_dict['title'] = row[1]
			temp_dict['description'] = row[2]
			if temp_dict['label'] == "1":
				text_set_1.append(temp_dict)
			elif temp_dict['label'] == "2":
				text_set_2.append(temp_dict)
			elif temp_dict['label'] == "3":
				text_set_3.append(temp_dict)
			elif temp_dict['label'] == "4":
				text_set_4.append(temp_dict)
			else:
				print('wrong')

	iteration = 0
	for text_list in [text_set_1,text_set_2,text_set_3,text_set_4]:
		for i in range(len(text_list)):
			text_dict[iteration] = text_list[i]
			iteration += 1
	
	corpus_list = [text_dict[k] for k,v in text_dict.items()]
	title_list, description_list = [], []
	corpus = []

	for i in range(len(corpus_list)):
		for word in corpus_list[i]['title'].split():
			word = word.lower()
			corpus.append(word)
		for word in corpus_list[i]['description'].split():
			word = word.lower()
			corpus.append(word)
		title_list.append(corpus_list[i]['title'])
		description_list.append(corpus_list[i]['description'])
	frequency = Counter(corpus)
	processed_dict = {}
	processed = []
	for k,v in frequency.items():
		if v > 4:
			processed_dict[k] = v
			processed.append(k)
	vocabulary = list(set(processed))

	total_gram_set = set()
	for i in range(len(corpus_list)):
		title_gram = gram_words(corpus_list[i]['title'], i, gram_mode)
		description_gram = gram_words(corpus_list[i]['description'], i, gram_mode)
		total_gram_set.update(title_gram)
		total_gram_set.update(description_gram)		
	total_gram_list = list(set(total_gram_set))
	total_gram_list = [word.split(',')[0][1:] + word.split(',')[1][:-1] for word in total_gram_list]

	ind2gram = {i: item for i, item in enumerate(total_gram_list)}
	gram2ind = {item: i for i, item in enumerate(total_gram_list)}



	W_emb, W_out = word2vec_trainer(processed, corpus_list, ind2gram, gram2ind, mode=mode, dimension=10, learning_rate=0.05)
	test_file = pd.read_csv('ag_news_csv/test.csv') 
	sim(test_file, gram2ind, W_emb, W_out)


main()


