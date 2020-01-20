import torch
from torch import nn
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchsummary import summary
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/gdrive')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epoch = 30
learning_rate = 0.0002
b1 = 0.5
b2 = 0.999
latent_vector = 100

print("device", device)
os.makedirs('generated-images-mnist', exist_ok = True)

# setting data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.conv1 = nn.ConvTranspose2d(latent_vector, 28*8, kernel_size = 7, stride = 1, padding = 0)
		self.batchnorm1 = nn.BatchNorm2d(28*8)
		self.conv2 = nn.ConvTranspose2d(28*8, 28*4, kernel_size = 4, stride = 2, padding = 1)
		self.batchnorm2 = nn.BatchNorm2d(28*4, 0.8)
		self.conv3 = nn.ConvTranspose2d(28*4, 1, kernel_size = 4, stride = 2, padding = 1)
		self.relu = nn.ReLU(inplace = True)
		self.tanh = nn.Tanh()


	def forward(self, data):
		x = data.view(batch_size, latent_vector, 1, 1)
		x = self.batchnorm1(self.relu(self.conv1(x))) #[128, 1024, 4, 4]
		x = self.batchnorm2(self.relu(self.conv2(x))) #[128, 512, 8, 8]
		x = self.relu(self.conv3(x)) #[128, 256, 16, 16]
		x = self.tanh(x)
		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.conv1 = nn.Conv2d(1, 28*4, kernel_size =4, stride = 2, padding = 1) #[128, 128, 16, 16]
		self.batchnorm1 = nn.BatchNorm2d(28*4)
		self.conv2 = nn.Conv2d(28*4, 28*8, kernel_size =4, stride = 2, padding = 1) #[128, 256, 8, 8]
		self.batchnorm2 = nn.BatchNorm2d(28*8)
		self.conv3 = nn.Conv2d(28*8, 1, kernel_size =7, stride = 1, padding = 0) #[128, 512, 4, 4]
		self.leakyrelu = nn.LeakyReLU(0.2, inplace = True)
		self.sigmoid = nn.Sigmoid()


	def forward(self, data):
		x = self.batchnorm1(self.leakyrelu(self.conv1(data)))
		x = self.batchnorm2(self.leakyrelu(self.conv2(x)))
		x = self.leakyrelu(self.conv3(x))
		x = self.sigmoid(x).squeeze(1).squeeze(1)
		return x

generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr = learning_rate, betas = (b1, b2))
d_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate, betas = (b1, b2))
criterion = nn.BCELoss()

# train
generator.train()
discriminator.train()

for iteration in range(epoch):
	for i, (images, labels) in enumerate(train_loader):
		random_vector = torch.randn(batch_size,latent_vector).to(device)
		labels = labels.to(device)
		real_images = images.to(device)
		if real_images.shape[0] < batch_size: break 

		generated_image = generator(random_vector)
		# # train discriminator		
		d_loss = criterion(discriminator(real_images).to(device), torch.ones(batch_size,1).to(device)) + criterion(discriminator(generated_image).to(device), torch.zeros(batch_size,1).to(device))
		d_optimizer.zero_grad()
		d_loss.backward(retain_graph = True)
		d_optimizer.step()
		

		# traain generator
		g_loss = criterion(discriminator(generated_image).to(device), torch.ones(batch_size,1).to(device))
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()
		
		loss = d_loss / (d_loss + g_loss)

		if i % 50 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(iteration+1, epoch, i, int(len(train_loader)), loss.item(), g_loss, d_loss))
			plot_list = []
			test = generated_image[0]
			npimg = test.detach().cpu().numpy().squeeze()
			plot_list.append(npimg)
			save_image(torch.tensor(plot_list), 'generated-images-mnist/'+str(epoch+1)+'-'+str(i)+'.png', nrow=1)
			plt.imshow(npimg, cmap = 'gray')
			plt.show()



