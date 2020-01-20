import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
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
parser = argparse.ArgumentParser(description = "Variational AutoEncoder using MNIST dataset")
parser.add_argument('--batch_size', type = int, default = 64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 10, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.001, help = "learning_rate")
parser.add_argument('--latent_dim', type = int, default = 100, help = "latent_vector dimension")
args = parser.parse_args()

print("device", device)
print(args)

os.makedirs('generated-images', exist_ok = True)


# setting data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = args.batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)


def image_show(img):
	npimg = img.numpy()
	plt.imshow(npimg, cmap = cm.gray)
	plt.show()

def loss_func(input, generated_data, mu, sigma):
	input = input.view(-1, 784)
	generated_data = torch.where(torch.isnan(generated_data), torch.zeros_like(generated_data), generated_data)
	loss1 = F.binary_cross_entropy(generated_data, input, size_average = False)
	loss2 = -0.5*torch.sum((1+torch.log(sigma**2) - mu**2 - sigma**2))
	return loss1+ loss2


class VAE(nn.Module):
	def __init__(self):
		super(VAE,self).__init__()

		self.encoder_layer = nn.Linear(784, 400)
		self.encoder_mu = nn.Linear(400, args.latent_dim)
		self.encoder_sigma = nn.Linear(400, args.latent_dim)

		self.decoder_layer1 = nn.Linear(args.latent_dim,400)
		self.decoder_layer2 = nn.Linear(400, 784)		
		
	def encode(self, input):
		x = F.relu(self.encoder_layer(input))
		mu = self.encoder_mu(x)
		sigma = self.encoder_sigma(x)
		epsilon = torch.randn_like(sigma)
		latent_vector = mu + torch.exp(0.5 * sigma) * epsilon
		
		param = [mu, sigma]
		return latent_vector, param

	def decode(self, latent_vector):
		output = F.relu(self.decoder_layer1(latent_vector))
		output = F.sigmoid(self.decoder_layer2(output))
		return output


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

# train
model.train()

for epoch in range(args.epoch):
	for i, (images, labels) in enumerate(train_loader):
		loss = 0
		images = images.to(device)
		images = images.view(-1,784)
		latent_vector, param = model.encode(images)
		output = model.decode(latent_vector)
		loss += loss_func(images, output, param[0], param[1])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i % 100 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.epoch, i, int(len(train_loader)), loss.item()))
		if i % 400 == 0:
			plot_list = []
			for iter in range(16):
				test = output[iter]
				test = test.view(1,28,28)
				npimg = test.detach().numpy()
				plot_list.append(npimg)
				save_image(torch.tensor(plot_list), 'generated-images/'+str(epoch)+'-'+str(i)+'.png', nrow=4)

# test
with torch.no_grad():
	img_list = []
	input = torch.randn(9,args.latent_dim)
	output = model.decode(input)
	output = output.view(9,1,28,28).clone().detach()
	grid_image = make_grid(output, nrow = 3)
	save_image(torch.tensor(grid_image), 'final_result.png')
