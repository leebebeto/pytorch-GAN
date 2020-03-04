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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setting args
parser = argparse.ArgumentParser(description = "Generative Adversarial Network using MNIST dataset")
parser.add_argument('--batch_size', type = int, default = 64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 8, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.0005, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")

args = parser.parse_args()


print("device", device)
print(args)
os.makedirs('generated-images', exist_ok = True)

# setting data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = args.batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train= False, transform = transforms.ToTensor()), shuffle = True)


class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.layer1 = nn.Linear(args.latent_vector, 256)
		self.layer2 = nn.Linear(256, 512)
		self.layer3 = nn.Linear(512,1024)
		self.layer4 = nn.Linear(1024, 784)
		self.sigmoid = nn.Sigmoid()
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.tanh = nn.Tanh()

	def forward(self, data):
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		x = self.tanh(self.layer4(x))
		return x

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.layer1 = nn.Linear(784, 512)
		self.layer2 = nn.Linear(512, 256)
		self.layer3 = nn.Linear(256, args.latent_vector)
		self.layer4 = nn.Linear(args.latent_vector,1)
		self.sigmoid = nn.Sigmoid()
		self.leakyrelu = nn.LeakyReLU(0.2)

	def forward(self, data):
		data = data.view(-1,784)
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		x = self.layer4(x)
		x = self.sigmoid(x)
		return x

generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
d_optimizer = optim.Adam(discriminator.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
criterion = nn.BCELoss()
# train
generator.train()
discriminator.train()

for epoch in range(args.epoch):
	for i, (images, labels) in enumerate(train_loader):
		labels = labels.to(device)
		random_vector = torch.randn(args.batch_size,args.latent_vector)
		generated_image = generator(random_vector).view(args.batch_size,1,28,28)
		real_images = images.to(device)

		# train discriminator
		if real_images.shape[0] < args.batch_size: break 
		d_loss = criterion(discriminator(real_images), torch.ones(args.batch_size,1)) + criterion(discriminator(generated_image), torch.zeros(args.batch_size,1))
		d_optimizer.zero_grad()
		d_loss.backward(retain_graph = True)
		d_optimizer.step()
		

		# train generator
		g_loss = criterion(discriminator(generated_image), torch.ones(args.batch_size,1))
		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()
		
		loss = d_loss / (d_loss + g_loss)


		if i % 100 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(epoch+1, args.epoch, i, int(len(train_loader)), loss.item(), g_loss, d_loss))
		if i % 200 == 0:
			plot_list = []
			for iter in range(16):
				test = generated_image[iter]
				test = test.view(1,28,28)
				npimg = test.detach().numpy()
				plot_list.append(npimg)
				save_image(torch.tensor(plot_list), 'generated-images/'+str(epoch)+'-'+str(i)+'.png', nrow=4)

with torch.no_grad():
	test_list = []
	for i in range(16):
		test_image = torch.randn(1, args.latent_vector)
		test_image = generator(test_image).view(1,28,28)
		test_image = test_image.detach().numpy()
		test_list.append(test_image)
	save_image(torch.tensor(test_list), 'final-test-image.png', nrow=4)




