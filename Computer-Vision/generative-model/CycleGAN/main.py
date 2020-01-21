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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting args
parser = argparse.ArgumentParser(description = "Cycle GAN using CIFAR-10 dataset")
parser.add_argument('--batch_size', type = int, default =64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 15, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.005, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--A_nc", type=int, default=3, help=" # of A channels")
parser.add_argument("--output_nc", type=int, default=3, help="# of output channels")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")
parser.add_argument("--recon_lambda", type= float, default=5.0, help="reconstruction lambda")
parser.add_argument("--cycle_lambda", type=int, default=10.0, help="cycle lambda")
args = parser.parse_args()

os.makedirs('generated-images', exist_ok = True)

# setting data
class PairedDataset(Dataset):

	def __init__(self, data_path, train, transform):
		self.train = train
		self.train_A_data =[]
		self.train_B_data =[]
		self.test_A_data =[]
		self.test_B_data =[]
		data_list = os.listdir(data_path)[1:]

		for data_type in data_list:
			print('data_type', data_type)
			sub_data_path = os.getcwd() + '/' + data_path + '/' + data_type
			if self.train:
				train_A_path = sub_data_path + '/' +'trainA'
				train_B_path = sub_data_path + '/' +'trainB'
				self.train_A_data += os.listdir(train_A_path)
				self.train_A_data = [str(train_A_path) + '/' + i for i in self.train_A_data]
				self.train_B_data += os.listdir(train_B_path)
				self.train_B_data = [str(train_B_path) + '/' + i for i in self.train_B_data]
			else:
				test_A_path = sub_data_path + '/' + 'testA'
				test_B_path = sub_data_path + '/' + 'testB'
				self.test_A_data += os.listdir(test_A_path)
				self.test_A_data = [str(test_A_path) + '/' + i for i in self.test_A_data]
				self.test_B_data += os.listdir(test_B_path)
				self.test_B_data = [str(test_A_path) + '/' + i for i in self.test_B_data]

		try:
			self.train_B_data = self.train_B_data[:len(self.train_A_data)]
			self.test_B_data = self.test_B_data[:len(self.test_A_data)]
		except:
			pass

		self.len = len(self.train_A_data) if self.train else len(self.test_A_data)

	def __getitem__(self, index):
		# print(index)
		A_data = self.train_A_data if self.train else self.test_A_data
		B_data = self.train_B_data if self.train else self.test_B_data

		A_data = '/'+'/'.join(A_data[index].split('/')[-15:])
		B_data = '/'+'/'.join(B_data[index].split('/')[-15:])

		A_img = cv2.imread(A_data)
		A_img = np.transpose(A_img,(2,1,0))

		B_img = cv2.imread(B_data)
		B_img = np.transpose(B_img,(2,1,0))

		return_set = (A_img, B_img) 
		return return_set

	def __len__(self):
		return self.len

train_data = PairedDataset('../../data/paired_data', train = True, transform = transforms.ToTensor())
test_data = PairedDataset('../../data/paired_data', train = False, transform = transforms.ToTensor())

train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True)
# 	batch_size = args.batch_size, shuffle = True)

# test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)

class ResidualBlock(nn.Module):
	def __init__(self, in_channels):
		super(ResidualBlock, self).__init__()

		conv_block = [ nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 0),
					   nn.InstanceNorm2d(in_channels),
					   nn.ReLU(inplace = True),
					   nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride= 1, padding = 0),
					   nn.InstanceNorm2d(in_channels)]

		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, model):
		return model + self.conv_block(model)


class Generator(nn.Module):
	def __init__(self, input_nc, output_nc, n_residual_blocks):
		super(Generator,self).__init__()

		self.down_layer1 = nn.Conv2d(3,64, kernel_size =7, padding = 1)
		self.down_layer2 = nn.Conv2d(64, 128, kernel_size =3,  stride=2, padding =1)
		self.down_layer3 = nn.Conv2d(128, 256, kernel_size =3,  stride=2, padding =1)
		# self.down_layer4 = nn.Conv2d(256, 512, kernel_size =3,  stride=2, padding =1)
		self.up_layer1 = nn.ConvTranspose2d(256,128, kernel_size = 3, stride=2)
		self.up_layer2 = nn.ConvTranspose2d(128,64, kernel_size =3, stride=2, padding =2)
		self.up_layer3 = nn.ConvTranspose2d(64, 3, kernel_size =6)
		# self.up_layer4 = nn.ConvTranspose2d(64,3, kernel_size =4,  stride=2, padding =1)
		self.instance_norm1 = nn.InstanceNorm2d(64)
		self.instance_norm2 = nn.InstanceNorm2d(128)
		self.instance_norm3 = nn.InstanceNorm2d(256)
		self.relu = nn.ReLU(inplace = True)
		self.tanh = nn.Tanh()

	def forward(self, data):
		x = self.relu(self.instance_norm1(self.down_layer1(data)))
		print('generator 1', x.shape)
		x = self.relu(self.instance_norm2(self.down_layer2(x)))
		print('generator 2', x.shape)
		x = self.relu(self.instance_norm3(self.down_layer3(x)))
		print('generator 3', x.shape)
		# residual_block = ResidualBlock(x.shape[1])
		# for i in range(9):
		# 	x += residual_block.forward(x)


		x = self.relu(self.instance_norm2(self.up_layer1(x)))
		print('generator 4', x.shape)
		x = self.relu(self.instance_norm1(self.up_layer2(x)))
		print('generator 5', x.shape)
		x = self.relu(self.up_layer3(x))
		print('generator 6', x.shape)
		x = self.tanh(x)
		print('generator 7', x.shape)
		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size =4, stride = 2, padding =1)
		self.conv2 = nn.Conv2d(64, 128, kernel_size =4, stride = 2, padding =1) 
		self.conv3 = nn.Conv2d(128, 256, kernel_size =4, stride = 2, padding =1)
		self.conv4 = nn.Conv2d(256, 512, kernel_size =4, stride = 2, padding =1) 
		self.fc = nn.Conv2d(512,1, kernel_size = 4, padding = 1)
		self.instancenorm1 = nn.InstanceNorm2d(64)
		self.instancenorm2 = nn.InstanceNorm2d(128)
		self.instancenorm3 = nn.InstanceNorm2d(256)
		self.instancenorm4 = nn.InstanceNorm2d(512)
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, data):
		print('1', data.shape)
		x = self.leakyrelu(self.instancenorm1(self.conv1(data)))
		print('2', x.shape)
		x = self.leakyrelu(self.instancenorm2(self.conv2(x)))
		print('3', x.shape)
		x = self.leakyrelu(self.instancenorm3(self.conv3(x)))
		print('4', x.shape)
		x = self.leakyrelu(self.instancenorm4(self.conv4(x)))
		print('5', x.shape)
		x = self.sigmoid(self.fc(x))
		print('6', x.shape)

		return x 

generator_A2B = Generator(args.A_nc, args.output_nc, 9).to(device)
generator_B2A = Generator(args.A_nc, args.output_nc, 9).to(device)
discriminator_A = Discriminator().to(device)
discriminator_B = Discriminator().to(device)

summary(generator_A2B, (3,256,256))
summary(discriminator_A, (3,256,256))
g_optimizer = optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()), lr = args.learning_rate, betas = (args.b1, args.b2))
d_A_optimizer = optim.Adam(discriminator_A.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
d_B_optimizer = optim.Adam(discriminator_B.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))

criterion_gan = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_reconstruction = nn.L1Loss()

generator_A2B.train()
generator_B2A.train()
discriminator_A.train()
discriminator_B.train()

for epoch in range(args.epoch):
	for i, (A_images, B_images) in enumerate(train_loader):

		real_A_images = A_images.float().to(device)
		real_B_images = B_images.float().to(device)

		if A_images.shape[0] < args.batch_size: break 

		g_optimizer.zero_grad()

		# reconstruction_loss
		post_real_A = generator_B2A(real_A_images)
		reconstruction_A_loss = criterion_reconstruction(real_A_images, post_real_A)
		post_real_B = generator_A2B(real_B_images)
		reconstruction_B_loss = criterion_reconstruction(real_B_images, post_real_B)
		
		reconstruction_loss = (reconstruction_A_loss + reconstruction_B_loss) / 2		

		# gan_loss
		post_generated_B = generator_A2B(real_A_images)
		gan_B_loss = criterion_gan(discriminator_B(post_generated_B), torch.ones(args.batch_size, discriminator_B(post_generated_B).shape[1], discriminator_B(post_generated_B).shape[2], discriminator_B(post_generated_B).shape[2]))
		post_generated_A = generator_B2A(real_B_images)
		gan_A_loss = criterion_gan(discriminator_A(post_generated_A), torch.ones(args.batch_size, discriminator_A(post_generated_A).shape[1], discriminator_A(post_generated_A).shape[2], discriminator_B(post_generated_B).shape[2]))

		gan_loss = (gan_A_loss + gan_B_loss) / 2
		# cycle_loss
		A2B = generator_A2B(real_A_images)	
		cycle_A_loss = criterion_cycle(real_B_images, A2B)
		B2A = generator_A2B(real_B_images)	
		cycle_B_loss = criterion_cycle(real_A_images, B2A)
		
		cycle_loss = (cycle_A_loss + cycle_B_loss) / 2
		# print('cycle_loss', cycle_loss)

		g_loss = args.recon_lambda * reconstruction_loss + gan_loss + args.cycle_lambda *cycle_loss
		print('g_loss', g_loss)

		g_loss.backward()
		g_optimizer.step()


		# real_loss
		# train discriminator		
		d_A_optimizer.zero_grad()
		disc_real_A = discriminator_A(real_A_images)
		d_A_loss_real = criterion_gan(disc_real_A, torch.ones(args.batch_size, disc_real_A.shape[1], disc_real_A.shape[2], disc_real_A.shape[3]))
		d_A_loss_fake = criterion_gan(discriminator_A(B2A), torch.zeros(args.batch_size, discriminator_A(B2A).shape[1], discriminator_A(B2A).shape[2], discriminator_A(B2A).shape[3]))
		d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
		d_A_loss.backward(retain_graph = True)
		d_A_optimizer.step()
		
		d_B_optimizer.zero_grad()
		disc_real_B = discriminator_B(real_B_images)
		d_B_loss_real = criterion_gan(disc_real_B, torch.ones(args.batch_size, disc_real_B.shape[1], disc_real_B.shape[2], disc_real_B.shape[3]))
		d_B_loss_fake = criterion_gan(discriminator_B(A2B), torch.zeros(args.batch_size, discriminator_B(A2B).shape[1], discriminator_B(A2B).shape[2], discriminator_B(A2B).shape[3]))
		d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
		d_B_loss.backward()
		d_B_optimizer.step()

		
		d_loss = d_A_loss + d_B_loss
		print('d_loss', d_loss)
		# if i % 50 == 0:	
		# 	print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(epoch+1, args.epoch, i, int(len(train_loader)), loss.item(), g_loss, d_loss))
		# 	plot_list = []
		# 	test = generated_images[0]
		# 	npimg = test.detach().numpy()
		# 	plot_list.append(npimg)
		# 	save_image(torch.tensor(plot_list), 'generated-images/'+str(epoch+1)+'-'+str(i)+'.png', nrow=1)


		# out_channel = 64
		# self.down_layer1 = nn.Conv2d(3, out_channel, kernel_size =7 , stride=2, padding =1)
		# self.down_layer2 = nn.Conv2d(out_channel, out_channel*2, kernel_size =3,  stride=2, padding =1)
		# self.down_layer3 = nn.Conv2d(out_channel*2, out_channel*4, kernel_size =3,  stride=2, padding =1)
		# self.down_layer4 = nn.Conv2d(2out_channel*4, out_channel*8, kernel_size =3,  stride=2, padding =1)
		# self.up_layer1 = nn.ConvTranspose2d(out_channel*8, out_channel*4, kernel_size = 2, stride=2, padding =0)
		# self.up_layer2 = nn.ConvTranspose2d(out_channel*4, out_channel*2, kernel_size =2, stride=2, padding =0)
		# self.up_layer3 = nn.ConvTranspose2d(out_channel*2, out_channel, kernel_size =2,  stride=2, padding =0)
		# self.up_layer4 = nn.ConvTranspose2d(out_channel*8, A_nc, kernel_size =4,  stride=2, padding =1)
		# self.instance_norm1 = nn.InstanceNorm2d(out_channel)
		# self.instance_norm2 = nn.InstanceNorm2d(out_channel * 2)
		# self.instance_norm3 = nn.InstanceNorm2d(out_channel * 4)
		# self.instance_norm4 = nn.InstanceNorm2d(out_channel * 8)
		# self.relu = nn.ReLU(inplace = True)
