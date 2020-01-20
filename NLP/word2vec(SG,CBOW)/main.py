import torch
import random
from random import shuffle
from collections import Counter
import argparse
import numpy as np

def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID+1 < len(corpus):
        context += corpus[wordID+1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)



def Skipgram(centerWord, contextWord, inputMatrix, outputMatrix):
    centerWordVector = []
    contextWordVector = []
    for i in range(inputMatrix.shape[0]):
        if i == centerWord:
            centerWordVector.append(1)
        if i == contextWord:
            contextWordVector.append(1)
        else:
            centerWordVector.append(0)
            contextWordVector.append(0)
    centerWordVector = torch.FloatTensor(centerWordVector)
    contextWordVector = torch.FloatTensor(contextWordVector)
    #feed forward 
    hidden_layer = torch.matmul(torch.t(inputMatrix),centerWordVector)
    output_layer = torch.matmul(torch.t(outputMatrix),hidden_layer)    
    e = torch.exp(output_layer)
    final_layer = e / torch.sum(e)

    loss = -torch.log(final_layer[contextWord]+1e-7)
    dfinal_layer = final_layer
    dfinal_layer[contextWord] -= 1
    grad_out = torch.from_numpy(np.outer(hidden_layer.numpy(), dfinal_layer.numpy()))    
    grad_emb = torch.from_numpy(np.outer(centerWordVector.numpy(), np.dot(outputMatrix,dfinal_layer.numpy().T)))
    return loss, grad_emb, grad_out


def CBOW(centerWord, contextWords, inputMatrix, outputMatrix):
    centerWordVector = []
    contextWordVector = []
    for i in range(inputMatrix.shape[0]):
        if i == centerWord:
            centerWordVector.append(1)
        else:
            centerWordVector.append(0)

    for i in range(inputMatrix.shape[0]):
        if i in contextWords:
            contextWordVector.append(1)
        else:
            contextWordVector.append(0)
    
    
    centerWordVector = torch.FloatTensor(centerWordVector)
    contextWordVector = torch.FloatTensor(contextWordVector)
    contextWordVector /= len(contextWords)

    hidden_layer = torch.matmul(torch.t(inputMatrix),contextWordVector)
    output_layer = torch.matmul(torch.t(outputMatrix),hidden_layer)
    e = torch.exp(output_layer)
    final_layer = e / torch.sum(e)

    loss = -(torch.log(final_layer[centerWord])+1e-7)
    dfinal_layer = final_layer
    dfinal_layer[centerWord] -= 1
    grad_out = torch.from_numpy(np.outer(hidden_layer.numpy(), dfinal_layer.numpy()))    
    grad_emb = torch.from_numpy(np.outer(centerWordVector.numpy(), np.dot(outputMatrix,dfinal_layer.numpy().T)))

    return loss, grad_emb, grad_out

def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=100, learning_rate=0.1, iteration=50):

# Xavier initialization of weight matrices
    W_emb = torch.randn(len(word2ind), dimension) / (dimension**0.5)
    W_out = torch.randn(dimension, len(word2ind)) / (dimension**0.5)
    window_size = 5
    losses=[]
    for i in range(iteration):
        #Training word2vec using SGD
        centerword, context = getRandomContext(corpus, window_size)
        centerInd = word2ind[centerword]
        contextInds = [word2ind[word] for word in context]
        if mode=="CBOW":
            L, G_emb, G_out = CBOW(centerInd, contextInds, W_emb, W_out)
            W_emb -= learning_rate*G_emb
            W_out -= learning_rate*G_out
            losses.append(L.item())

        elif mode=="SG":
            for contextInd in contextInds:
                L, G_emb, G_out = Skipgram(centerInd, contextInd, W_emb, W_out)
                W_emb -= learning_rate*G_emb
                W_out -= learning_rate*G_out
                losses.append(L.item())
                
        else:
            print("Unknown mode : "+mode)
            exit()

        if i%500==0:
            print(i)
            avg_loss=sum(losses)/len(losses)
            print("Loss : %f" %(avg_loss,))
            losses=[]

    return W_emb, W_out


# simple function for printing similar words 
def sim(testword, word2ind, ind2word, matrix):
    length = (matrix*matrix).sum(1)**0.5
    wi = word2ind[testword]
    inputVector = matrix[wi].reshape(1,-1)/length[wi]
    sim = (inputVector@matrix.t())[0]/length
    values, indices = sim.squeeze().topk(5)
    
    print()
    print("===============================================")
    print("The most similar words to \"" + testword + "\"")
    for ind, val in zip(indices,values):
        print(ind2word[ind.item()]+":%.3f"%(val,))
    print("===============================================")
    print()


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('--mode', metavar='mode', type=str, default = 'CBOW',
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('--part', metavar='part', type=str, default = 'part', 
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default = 0.001,
                        help='learing rate')
    parser.add_argument('--iteration', metavar='iteration', type=int, default = 30000,
                        help='iteration')
    parser.add_argument('--dimension', metavar='dimension', type=int, default = 64,
                        help='dimension')

    args = parser.parse_args()
    mode = args.mode
    part = args.part
    learning_rate = args.learning_rate
    iteration = args.iteration
    dimension = args.dimension



    #Load and tokenize corpus
    print("loading...")
    if part=="part":
        text = open('text8.txt',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8.txt',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    corpus = text.split()
    frequency = Counter(corpus)
    processed = []
    #Discard rare words
    for word in corpus:
        if frequency[word]>4:
            processed.append(word)
    vocabulary = set(processed)
    #Assign an index number to a word
    word2ind = {}
    word2ind[" "]=0
    i = 1
    for word in vocabulary:
        word2ind[word] = i
        i+=1

    ind2word = {}
    for k,v in word2ind.items():
        ind2word[v]=k

    print("Vocabulary size")
    print(len(word2ind))
    print()

    #Training section
    emb,_ = word2vec_trainer(processed, word2ind, mode=mode, dimension= dimension, learning_rate= learning_rate, iteration= iteration)
    
    #Print similar words
    testwords = ["one", "are", "he", "have", "many", "first", "all", "world", "people", "after"]
    for tw in testwords:
        sim(tw,word2ind,ind2word,emb)

main()