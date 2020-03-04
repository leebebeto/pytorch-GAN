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

		conv_block = [ nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 1),
					   nn.InstanceNorm2d(in_channels),
					   nn.ReLU(inplace = True),
					   nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 1),
					   nn.InstanceNorm2d(in_channels)]
		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, x):
		return x + self.conv_block(x)


class Generator(nn.Module):
	def __init__(self, n_residual_blocks, out_nc, n_attribute):
		super(Generator,self).__init__()


		self.down_layer1 = nn.Sequential(nn.Conv2d(3 + n_attribute, out_nc, kernel_size =7, stride =1, padding = 3, bias = False),
									nn.InstanceNorm2d(out_nc, affine= True, track_running_stats=True),
									nn.ReLU(inplace = True))
		self.down_layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.InstanceNorm2d(out_nc * 2, affine= True, track_running_stats=True),
									nn.ReLU(inplace = True))
		self.down_layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =4, stride = 2, padding = 1, bias =False),
									nn.InstanceNorm2d(out_nc * 4, affine= True, track_running_stats=True),
									nn.ReLU(inplace = True))

		self.up_layer1 = nn.Sequential(nn.ConvTranspose2d(out_nc * 4, out_nc * 2, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.InstanceNorm2d(out_nc * 2, affine=True, track_running_stats=True),
									nn.ReLU(inplace = True))
		self.up_layer2 = nn.Sequential(nn.ConvTranspose2d(out_nc * 2, out_nc, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.InstanceNorm2d(out_nc, affine=True, track_running_stats=True),
									nn.ReLU(inplace = True))
		self.up_layer3 = nn.Sequential(nn.ConvTranspose2d(out_nc , 3, kernel_size =7, stride = 1, padding = 3))

		self.residual_block = ResidualBlock(256)
		self.tanh = nn.Tanh()

	def forward(self, image, attr):
		attr = attr.unsqueeze(2).unsqueeze(3)
		attr = attr.repeat(1,1, image.shape[2], image.shape[3])
		data = torch.cat((image, attr), 1)
		x = self.down_layer1(data)
		x = self.down_layer2(x)
		x = self.down_layer3(x)
		
		for i in range(6):
			x = self.residual_block(x)

		x = self.up_layer1(x)
		x = self.up_layer2(x)
		x = self.up_layer3(x)
		x = self.tanh(x)
		return x

class Discriminator(nn.Module):
	def __init__(self, out_nc, n_domain):
		super(Discriminator,self).__init__()
		self.n_domain = n_domain
		self.layer1 = nn.Sequential(nn.Conv2d(3, out_nc, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))

		self.layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))

		self.layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))	
									

		self.layer4 = nn.Sequential(nn.Conv2d(out_nc * 4, out_nc * 8, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))
		
		self.layer5 = nn.Sequential(nn.Conv2d(out_nc * 8, out_nc * 16, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))	
									

		self.layer6 = nn.Sequential(nn.Conv2d(out_nc * 16, out_nc * 32, kernel_size =4, stride = 2, padding = 1),
									nn.LeakyReLU(0.01))
		
		self.layer7_src =  nn.Conv2d(out_nc * 32, 1, kernel_size =3, stride = 1, padding = 1, bias = False)
			
		self.layer7_cls =  nn.Conv2d(out_nc * 32, n_domain, kernel_size =2, stride = 1, padding = 0, bias = False)

	def forward(self, data):
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.layer6(x)
		src_pred = self.layer7_src(x)
		cls_pred = self.layer7_cls(x)
		cls_pred = cls_pred.view(cls_pred.size(0), -1)

		return src_pred, cls_pred 

