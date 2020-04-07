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


class STU(nn.Module):
	def __init__(self, in_dim, out_dim, n_attr):
		super(STU, self).__init__()
		self.transpose_attr = nn.ConvTranspose2d(in_dim * 2 + n_attr, out_dim, 4, 2, 1, bias = False)
		self.transpose = nn.ConvTranspose2d(in_dim * 2, out_dim, 4, 2, 1, bias = False)


		self.reset_gate = nn.Sequential(nn.Conv2d(in_dim + out_dim, out_dim, 3, 1, 1, bias = False),
										nn.BatchNorm2d(out_dim),
										nn.Sigmoid())

		self.update_gate = nn.Sequential(nn.Conv2d(in_dim + out_dim, out_dim, 3, 1, 1, bias = False),
										nn.BatchNorm2d(out_dim),
										nn.Sigmoid())

		self.hidden_gate = nn.Sequential(nn.Conv2d(in_dim + out_dim, out_dim, 3, 1, 1, bias = False),
										nn.BatchNorm2d(out_dim),
										nn.Tanh())

	def forward(self, stu_input, f_l):
		if f_l.shape[3] == 8:
			s_l_prev = self.transpose_attr(stu_input)
		else:
			s_l_prev = self.transpose(stu_input)
		r_l = self.reset_gate(torch.cat((s_l_prev, f_l), 1)) 
		z_l = self.update_gate(torch.cat((s_l_prev, f_l), 1))
		s_l = r_l * s_l_prev
		f_hat = self.hidden_gate(torch.cat((f_l,s_l), 1))
		f_l_final = (1-z_l) * s_l_prev + z_l * f_hat
		return s_l, f_l_final

class Encoder(nn.Module):
	def __init__(self, out_nc, n_attr):
		super(Encoder, self).__init__()
		self.down_layer1 = nn.Sequential(nn.Conv2d(3, out_nc, kernel_size =4, stride =2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc),
									nn.LeakyReLU(0.2, inplace=True))
		self.down_layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc * 2),
									nn.LeakyReLU(0.2, inplace=True))
		self.down_layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =4, stride = 2, padding = 1, bias =False),
									nn.BatchNorm2d(out_nc * 4),
									nn.LeakyReLU(0.2, inplace=True))
		self.down_layer4 = nn.Sequential(nn.Conv2d(out_nc * 4, out_nc * 8, kernel_size =4, stride = 2, padding = 1, bias =False),
									nn.BatchNorm2d(out_nc * 8),
									nn.LeakyReLU(0.2, inplace=True))
		self.down_layer5 = nn.Sequential(nn.Conv2d(out_nc * 8, out_nc * 16, kernel_size =4, stride = 2, padding = 1, bias =False),
									nn.BatchNorm2d(out_nc * 16),
									nn.LeakyReLU(0.2, inplace=True))
		self.stu1 = STU(out_nc, out_nc, n_attr).to(device)
		self.stu2 = STU(out_nc * 2, out_nc * 2, n_attr).to(device)
		self.stu3 = STU(out_nc * 4, out_nc * 4, n_attr).to(device)
		self.stu4 = STU(out_nc * 8, out_nc * 8, n_attr).to(device)

	def forward(self, image, diff_attr):
		x1 = self.down_layer1(image)
		x2 = self.down_layer2(x1)
		x3 = self.down_layer3(x2)
		x4 = self.down_layer4(x3)
		x5 = self.down_layer5(x4)

		diff_attr = diff_attr.unsqueeze(2).unsqueeze(3)

		feature_list = []
		for i, (x, stu) in enumerate(zip([x4, x3, x2, x1], [self.stu4, self.stu3, self.stu2, self.stu1])):
			if i ==0 :
				diff_attr = diff_attr.expand(diff_attr.shape[0], diff_attr.shape[1], x5.shape[2], x5.shape[3])
				stu_input = torch.cat((x5, diff_attr), 1)
				stu_result, feature_result  = stu(stu_input, x)
			else:
				stu_input, feature_input = stu_result, feature_result
				stu_result, feature_result  = stu(stu_input, x)
			
			feature_list.append(stu_result)

		return x5, feature_list


class Decoder(nn.Module):
	def __init__(self, out_nc, n_attr):
		super(Decoder, self).__init__()
		self.up_layer1 = nn.Sequential(nn.ConvTranspose2d(1024 + n_attr, out_nc * 16, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc * 16),
									nn.ReLU(inplace = True))
		self.up_layer2 = nn.Sequential(nn.ConvTranspose2d(out_nc * 16 + out_nc * 8, out_nc * 8, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc * 8),
									nn.ReLU(inplace = True))
		self.up_layer3 = nn.Sequential(nn.ConvTranspose2d(out_nc * 8 + out_nc * 4, out_nc * 4, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc * 4),
									nn.ReLU(inplace = True))
		self.up_layer4 = nn.Sequential(nn.ConvTranspose2d(out_nc * 4 + out_nc * 2, out_nc * 2, kernel_size =4, stride = 2, padding = 1, bias = False),
									nn.BatchNorm2d(out_nc * 2),
									nn.ReLU(inplace = True))
		self.up_layer5 = nn.Sequential(nn.ConvTranspose2d(out_nc * 2 + out_nc , 3, kernel_size =4, stride = 2, padding = 1),
									nn.Tanh())

	def forward(self, latent_vector, diff_attr, feature_list):
		diff_attr = diff_attr.unsqueeze(2).unsqueeze(3).expand(diff_attr.shape[0], diff_attr.shape[1], latent_vector.shape[2], latent_vector.shape[3])
		latent_vector = torch.cat((latent_vector, diff_attr), 1)
		x = self.up_layer1(latent_vector)
		x = torch.cat((x, feature_list[0]), 1)
		x = self.up_layer2(x)
		x = torch.cat((x, feature_list[1]), 1)
		x = self.up_layer3(x)
		x = torch.cat((x, feature_list[2]), 1)
		x = self.up_layer4(x)
		x = torch.cat((x, feature_list[3]), 1)
		x = self.up_layer5(x)
		return x 


class Generator(nn.Module):
	def __init__(self, out_nc, n_attr):
		super(Generator,self).__init__()
		self.encoder = Encoder(out_nc, n_attr)
		self.decoder = Decoder(out_nc, n_attr)


	def forward(self, image, diff_attr):
		latent_vector, feature_list = self.encoder(image, diff_attr)
		x = self.decoder(latent_vector, diff_attr, feature_list)
		return x

class Discriminator(nn.Module):
	def __init__(self, out_nc, n_attr):
		super(Discriminator,self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(3, out_nc, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc),
									nn.LeakyReLU(0.2, inplace=True))

		self.layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc * 2, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 2),
									nn.LeakyReLU(0.2, inplace=True))

		self.layer3 = nn.Sequential(nn.Conv2d(out_nc * 2, out_nc * 4, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 4),
									nn.LeakyReLU(0.2, inplace=True))	
									

		self.layer4 = nn.Sequential(nn.Conv2d(out_nc * 4, out_nc * 8, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 8),
									nn.LeakyReLU(0.2, inplace=True))
		
		self.layer5 = nn.Sequential(nn.Conv2d(out_nc * 8, out_nc * 16, kernel_size =4, stride = 2, padding = 1),
									nn.InstanceNorm2d(out_nc * 16),
									nn.LeakyReLU(0.2, inplace=True))	
									

		self.layer6_src =  nn.Sequential(nn.Linear(out_nc * 16 * 16, out_nc * 16),
										 nn.LeakyReLU(0.2, inplace=True))
			
		self.layer6_cls =  nn.Sequential(nn.Linear(out_nc * 16 * 16, out_nc * 16),
										 nn.LeakyReLU(0.2, inplace=True))

		self.layer7_src =  nn.Linear(out_nc * 16, 1)

		self.layer7_cls =  nn.Sequential(nn.Linear(out_nc * 16, n_attr),
										 nn.Sigmoid())

	def forward(self, data):
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = x.view(x.shape[0], -1)
		src_pred = self.layer6_src(x)
		src_pred = self.layer7_src(src_pred)
		cls_pred = self.layer6_cls(x)
		cls_pred = self.layer7_cls(cls_pred)

		return src_pred, cls_pred 

