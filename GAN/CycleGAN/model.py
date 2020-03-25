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
from PIL import Image
import os
from operator import itemgetter
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
import data

# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
	def __init__(self, in_channels):
		super(ResidualBlock, self).__init__()

		conv_block = [ nn.ReflectionPad2d(1),
					   nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 0),
					   nn.InstanceNorm2d(in_channels),
					   nn.ReLU(inplace = True),
					   nn.ReflectionPad2d(1),
					   nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 0),
					   nn.InstanceNorm2d(in_channels)]

		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, model):
		return model + self.conv_block(model)


class Generator(nn.Module):
	def __init__(self, out_nc,  n_residual_blocks):
		super(Generator,self).__init__()


		self.down_layer1 = nn.Sequential(
									nn.ReflectionPad2d(3),
									nn.Conv2d(3, out_nc, kernel_size =7),
									nn.InstanceNorm2d(out_nc),
									nn.ReLU(inplace = True))
		self.down_layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =3, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 2),
									nn.ReLU(inplace = True))
		self.down_layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =3, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 4),
									nn.ReLU(inplace = True))

		self.up_layer1 = nn.Sequential(
									nn.Upsample(scale_factor = 2),
									nn.Conv2d(out_nc * 4, out_nc * 2, kernel_size =3, stride =1 , padding = 1),
									nn.InstanceNorm2d(out_nc * 2),
									nn.ReLU(inplace = True))
		self.up_layer2 = nn.Sequential(
									nn.Upsample(scale_factor = 2),
									nn.Conv2d(out_nc * 2, out_nc, kernel_size =3, stride =1 , padding = 1),
									nn.InstanceNorm2d(out_nc),
									nn.ReLU(inplace = True))

		self.up_layer3 = nn.Sequential(
									nn.ReflectionPad2d(3),
									nn.Conv2d(out_nc , 3, kernel_size =7),
									nn.Tanh())

		self.residual_block = ResidualBlock(256).to(device)

		self.tanh = nn.Tanh()

	def forward(self, data):
		x = self.down_layer1(data)
		x = self.down_layer2(x)
		x = self.down_layer3(x)
		for i in range(9):
			x = self.residual_block(x)
		x = self.up_layer1(x)
		x = self.up_layer2(x)
		x = self.up_layer3(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, out_nc):
		super(Discriminator,self).__init__()

		self.layer1 = nn.Sequential(nn.Conv2d(3, out_nc, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc),
									nn.ReLU(inplace = True))

		self.layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 2),
									nn.ReLU(inplace = True))

		self.layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 4),
									nn.ReLU(inplace = True))

		self.layer4 = nn.Sequential(nn.Conv2d(out_nc * 4, out_nc * 8, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 8),
									nn.ReLU(inplace = True))

		self.layer5 = nn.Sequential(
						nn.ZeroPad2d((1,0,1,0)),
						nn.Conv2d(out_nc * 8, 1, kernel_size = 4, padding = 1))
		
		self.sigmoid = nn.Sigmoid()

	def forward(self, data):
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.sigmoid(x)

		return x 

