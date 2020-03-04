import torch
from torch import nn
from torch import optim
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting args
batch_size = 512
epoch = 150
learning_rate_g = 0.002
learning_rate_d = 0.002
b1 = 0.5
b2 = 0.999
latent_vector = 100
out_nc = 64
parallel_devices = 2
print("device", device)
os.makedirs('generated-images-cifar-128', exist_ok = True)

transforms_ = transforms.Compose([
	transforms.Resize(64),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# setting data
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../../data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)


class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()

		self.layer1 = nn.Sequential(nn.ConvTranspose2d(latent_vector, out_nc * 8, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 8),
					  nn.ReLU(inplace = True))

		self.layer2 = nn.Sequential(nn.ConvTranspose2d(out_nc * 8, out_nc * 4, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 4),
					  nn.ReLU(inplace = True))

		self.layer3 = nn.Sequential(nn.ConvTranspose2d(out_nc * 4, out_nc * 2, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 2),
					  nn.ReLU(inplace = True))


		self.layer4 = nn.Sequential(nn.ConvTranspose2d(out_nc * 2, out_nc, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc),
					  nn.ReLU(inplace = True))


		self.layer5 = nn.Sequential(nn.ConvTranspose2d(out_nc, 3, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(3),
					  nn.Tanh())

	def forward(self, data):
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()

		self.layer1 = nn.Sequential(nn.Conv2d(3, out_nc, kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.LeakyReLU(0.2, inplace = True))

		self.layer2 = nn.Sequential(nn.Conv2d(out_nc, out_nc*2,kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 2),
					  nn.LeakyReLU(0.2, inplace = True))

		self.layer3 = nn.Sequential(nn.Conv2d(out_nc*2, out_nc*4,kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 4),
					  nn.LeakyReLU(0.2, inplace = True))


		self.layer4 = nn.Sequential(nn.Conv2d(out_nc*4 out_nc*8,kernel_size = 4, stride = 2, padding = 1, bias= False),
					  nn.BatchNorm2d(out_nc * 8),
					  nn.LeakyReLU(0.2, inplace = True))

		self.layer5 = nn.Linear(out_nc * 64 * 2, 1)

		self.sigmoid = nn.Sigmoid()


	def forward(self, data):
		x = self.layer1(data)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		x = self.sigmoid(x)
		return x

generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr = learning_rate_g, betas = (b1, b2))
d_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate_d, betas = (b1, b2))
criterion = nn.BCELoss()

# train
generator.train()
discriminator.train()

for iteration in range(epoch):
	for i, (images, labels) in enumerate(train_loader):
		random_vector = torch.randn(batch_size,latent_vector,1,1).to(device)
		labels = labels.to(device)
		real_images = images.to(device)
		generated_image = generator(random_vector).to(device)
		if real_images.shape[0] < batch_size: break 
		
		valid = torch.ones(discriminator(real_images).shape[0], discriminator(real_images).shape[1]).to(device)
		fake = torch.zeros(discriminator(generated_image).shape[0],discriminator(generated_image).shape[1]).to(device)
		fake_real = torch.ones(discriminator(generated_image).shape[0],discriminator(generated_image).shape[1]).to(device)
		# train discriminator
		d_loss = criterion(discriminator(real_images).to(device), valid) + criterion(discriminator(generated_image).to(device), fake)
		d_optimizer.zero_grad()
		d_loss.backward(retain_graph = True)
		d_optimizer.step()
		

		# train generator
		g_loss = criterion(discriminator(generated_image).to(device), fake_real)
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()
		
		loss = d_loss / (d_loss + g_loss)

		if i % 50 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(iteration+1, epoch, i, int(len(train_loader)), loss.item(), g_loss, d_loss))	
		if i % 50 == 0:
			plot_list = []
			test = generated_image[0]
			npimg = test.detach().cpu().numpy()
			plot_list.append(npimg)
			npimg = np.rollaxis(npimg[::-1],0,3)
			save_image(torch.tensor(plot_list), 'generated-images-cifar-128/'+str(iteration+1)+'-'+str(i)+'.png', nrow=1)





