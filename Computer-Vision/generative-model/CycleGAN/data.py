import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchsummary import summary
import itertools
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import cv2
from operator import itemgetter
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary


# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting data
class PairedDataset(Dataset):

	def __init__(self, data_path, train, transform):
		self.train = train

		self.train_A_path = os.path.join(os.getcwd(), data_path, 'trainA')
		self.train_B_path = os.path.join(os.getcwd(), data_path, 'trainB')
		self.test_A_path = os.path.join(os.getcwd(), data_path, 'testA')
		self.test_B_path = os.path.join(os.getcwd(), data_path, 'testB')


		self.train_A = os.listdir(os.path.join(os.getcwd(), data_path, 'trainA'))
		self.train_B = os.listdir(os.path.join(os.getcwd(), data_path, 'trainB'))
		self.test_A = os.listdir(os.path.join(os.getcwd(), data_path, 'testA'))
		self.test_B = os.listdir(os.path.join(os.getcwd(), data_path, 'testB'))


		self.transform = transform
		self.len = len(self.train_A_path) if self.train else len(self.test_A_path)

	def __getitem__(self, index):
		
		A_data_path = self.train_A if self.train else self.test_A
		B_data_path = self.train_B if self.train else self.test_B

		A_data = os.path.join(self.train_A_path, A_data_path[(index-1) % len(self.train_A)])
		B_data = os.path.join(self.train_B_path, B_data_path[(index+1) % len(self.train_B)])

		A_img = Image.open(A_data)
		B_img = Image.open(B_data)
		
		A_img = self.transform(A_img)
		B_img = self.transform(B_img)
		return_set = (A_img, B_img) 
		return return_set

	def __len__(self):
		return self.len

# 	batch_size = args.batch_size, shuffle = True)

# test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)

