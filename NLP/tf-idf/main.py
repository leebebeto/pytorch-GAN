import nltk
import os 
import re
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
import math
import operator

# ------------- basic setting -----------------------
file_root = os.path.dirname(os.path.abspath(__file__))
with open(file_root+'/query.txt', 'r') as f:
	query = [line.strip() for line in f][0]

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer("english")

query = query.split(' ')
query = [snowball_stemmer.stem(i) for i in query]
data = os.listdir(file_root+'/data') 
N = len(data) + 1
data_dict = {}
stat_dict ={}
english = re.compile('[a-zA-Z]+')
stop = set(stopwords.words('english'))
#  --------------------------------------------------

def tokenize(data):
	# tokenize documents
	for datum in data:
		file_name = file_root+'/data/'+datum
		with open(file_name,'rb') as f:
			temp = [line.strip() for line in f]
			result = []		
			for sentence in temp:
				sentence = str(sentence.lower())
				tokens = nltk.word_tokenize(sentence)
			tokens = [i for i in tokens if i not in stop and english.match(i)]
			tokens = [i.replace("\\","") for i in tokens]
			tokens = [snowball_stemmer.stem(i) for i in tokens]
			text = nltk.Text(tokens)
		word_dict = {}
		stat_dict[datum] = text
		fdst = FreqDist(stat_dict[datum])
		vocab = fdst.most_common()
		for word in vocab:
			word_dict[word[0]] = word[1]
		data_dict[datum] = word_dict

	# tokenize query 
	word_dict = {}
	stat_dict['query.txt'] = query
	fdst = FreqDist(stat_dict['query.txt'])
	vocab = fdst.most_common()
	for word in vocab:
		word_dict[word[0]] = word[1]
	data_dict['query.txt'] = word_dict
	print('Index Construction: ', data_dict)	

	return data_dict


def tf_idf(data_dict, query):
	inverted_word = []	
	inverted_index = {} 
	document_list = [] 
	result_dict = {}	
	total_result = {}

	for word in data_dict:
		inverted_word.extend(list(data_dict[word]))
	inverted_word = list(set(inverted_word))
	inverted_word.sort()

	for word in inverted_word:
		temp = []
		for document in data_dict:
			if word in data_dict[document]:
				temp.append(document)
		inverted_index[word] = temp
	print('Inverted Index Construction: ', inverted_index)	

	for word in query:
		document_list.extend(inverted_index[word])
	# adding query text as document
	document_list = list(set(document_list))
	document_list.remove('query.txt')
	document_list = [int(doc[:-4]) for doc in document_list]
	document_list.sort()
	document_list = [str(doc)+'.txt' for doc in document_list]	
	document_noquery_list = document_list
	document_list.append('query.txt')

	print('Document list with query: ', document_list)	

	result_dict = {} 
	result_df = pd.DataFrame()

	for doc in document_list:	
		tf_idf_dict = {}
		current_doc_dict = data_dict[doc]
		temp_data_dict = {}
		for word in inverted_word:
			try:
				tf = current_doc_dict[word]
				df = len(inverted_index[word])
				w = math.log(1+tf)*math.log10(N/df)
			except:
				w = 0
			temp_data_dict[word] = w

		tf_idf_dict[doc] = temp_data_dict
		result_dict[doc] = tf_idf_dict.values()
	for i in result_dict.keys():
		for j in result_dict[i]:
			result_df['word'] = j
			result_df[i] = j.values()
	result_df = result_df.T
	result_df.to_csv('result_df.csv')
	result_np = result_df.as_matrix()
	result_np = np.delete(result_np, 0,0)
	y = result_np[len(result_np)-1]
	final_dict = {}
	for i in range(result_np.shape[0]-1):
		score = cos_similar(result_np[i],y)
		final_dict[document_noquery_list[i]] = score
	final_result = list(reversed(sorted(final_dict.items(), key=operator.itemgetter(1))))
	print('Total: ', final_result)
	final_result = final_result[0:5]
	final_result = [i[0] for i in final_result]
	print('Top 5: ', final_result)


def cos_similar(x,y):
	normal_x = np.linalg.norm(np.square(x))
	normal_y = np.linalg.norm(np.square(y))
	return np.dot(x,y)/(normal_x*normal_y)


def main():
	data_dict = tokenize(data)
	tf_idf(data_dict, query)	

if __name__ == "__main__":
	main()	
