import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
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
import re

# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting data
class CelebADataset(Dataset):

	def __init__(self, data_path, transforms_, mode):
		attributes = ['Bangs', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Young']
		self.image_path = os.getcwd() + '/' + data_path + '/images/'
		self.image_list = os.listdir(self.image_path)
		attr_file = os.getcwd() + '/' + data_path + '/list_attr_celeba.txt'
		self.mode = True if mode == "train" else "val"
		
		" referred to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/datasets.py"
		self.image2attribute = {}
		self.label_data = [line.rstrip() for line in open(attr_file, 'r')]
		self.label_list = self.label_data[1].split()
		for i, line in enumerate(self.label_data[2:]):
			image_name, *values = line.split()
			label_temp = []
			for attr in attributes:
				index = self.label_list.index(attr)	
				label_temp.append(1 * (values[index] == "1"))
			self.image2attribute[image_name] = label_temp
			
		self.transform = transforms_
		self.len = len(self.image2attribute.items())

	def __getitem__(self, index):
		image = Image.open(self.image_path + self.image_list[index])
		
		image = self.transform(image)
		attr = self.image2attribute[self.image_list[index]]
		attr = torch.FloatTensor(attr)
		return image, attr

	def __len__(self):
		return self.len
		
