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


# test_image부터 시도

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setting args
parser = argparse.ArgumentParser(description = "Generative Adversarial Network using MNIST dataset")
parser.add_argument('--batch_size', type = int, default = 64, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 15, help = "epoch")
parser.add_argument('--learning_rate', type = float, default = 0.001, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")
parser.add_argument("--out_layer", type=int, default=256, help="number of out layer")

args = parser.parse_args()


print("device", device)
print(args)
os.makedirs('generated-images', exist_ok = True)

# setting data
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= True, download = True, transform = transforms.ToTensor()),
	batch_size = args.batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data', train= False, transform = transforms.ToTensor()), shuffle = True)


class Generator(nn.Module):
	def __init__(self, out_layer):
		super(Generator,self).__init__()
		self.layer1 = nn.Linear(args.latent_vector + 10, out_layer)
		self.layer2 = nn.Linear(out_layer, out_layer * 2)
		self.layer3 = nn.Linear(out_layer * 2, out_layer * 4)
		self.layer4 = nn.Linear(out_layer * 4, 784)
		self.sigmoid = nn.Sigmoid()
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.tanh = nn.Tanh()

	def forward(self, data, condition):
		data = data.view(data.shape[0], -1)
		condition = condition.float()
		data = torch.cat((data,  condition), 1)
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		x = self.tanh(self.layer4(x))
		return x

class Discriminator(nn.Module):
	def __init__(self, out_layer):
		super(Discriminator,self).__init__()
		self.embedding = nn.Embedding(10,10)
		self.layer1 = nn.Linear(784 + 10, out_layer * 2)
		self.layer2 = nn.Linear(out_layer * 2, out_layer)
		self.layer3 = nn.Linear(out_layer, out_layer)
		self.layer4_cls = nn.Linear(out_layer, 10)
		self.layer4_valid = nn.Linear(out_layer, 1)
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax()
		self.leakyrelu = nn.LeakyReLU(0.2)

	def forward(self, data, condition):
		data = data.view(data.shape[0], -1)
		condition = condition.float()
		data = torch.cat((data, condition), 1)
		x = self.leakyrelu(self.layer1(data))
		x = self.leakyrelu(self.layer2(x))
		x = self.leakyrelu(self.layer3(x))
		cls = self.softmax(self.layer4_cls(x))
		validity = self.sigmoid(self.layer4_valid(x))
		return validity, cls

def label2vec(labels):
	result = torch.zeros(labels.shape[0], 10).long()
	for i in range(result.shape[0]):
		result[i][labels[i].item()] = 1.0
	return result

def valid(data):
	return torch.ones(data.size())

def fake(data):
	return torch.zeros(data.size())

generator = Generator(args.out_layer).to(device)
discriminator = Discriminator(args.out_layer).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
d_optimizer = optim.Adam(discriminator.parameters(), lr = args.learning_rate, betas = (args.b1, args.b2))
criterion_adv = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()

# train
generator.train()
discriminator.train()

for epoch in range(args.epoch):
	for i, (images, labels) in enumerate(train_loader):
		labels = labels.to(device)
		random_vector = torch.randn(args.batch_size,args.latent_vector).to(device)
		condition_vector = label2vec(labels).to(device)
		real_images = images.to(device)

		if real_images.shape[0] < args.batch_size: break 
		generated_image = generator(random_vector, condition_vector).view(args.batch_size,1,28,28).to(device)

		real_validity, real_cls = discriminator(real_images, condition_vector) 
		fake_validity, fake_cls = discriminator(generated_image, condition_vector)

		# train discriminator		
		d_loss_adv_real = criterion_adv(real_validity, valid(real_validity)) 
		d_loss_adv_fake = criterion_adv(fake_validity, fake(fake_validity))
		d_loss_adv = d_loss_adv_real + d_loss_adv_fake

		d_loss_cls = criterion_cls(real_cls, labels)

		d_loss = d_loss_adv + d_loss_cls

		d_optimizer.zero_grad()
		d_loss.backward(retain_graph = True)
		d_optimizer.step()
		

		# train generator
		g_loss_adv = criterion_adv(fake_validity, valid(fake_validity))
		g_loss_cls = criterion_cls(fake_cls, labels)
		g_loss = g_loss_adv + g_loss_cls

		g_optimizer.zero_grad()
		g_loss.backward()
		g_optimizer.step()
		
		loss = d_loss / (d_loss + g_loss)


		if i % 100 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, G-ADV-Loss: {:.4f}, G-CLS-Loss: {:.4f}, D-Loss: {:.4f}, D-ADV-Loss: {:.4f}, D-CLS-Loss: {:.4f} '.format(epoch+1, args.epoch, i, int(len(train_loader)), loss.item(), g_loss, g_loss_adv, g_loss_cls, d_loss, d_loss_adv, d_loss_cls))
		if i % 400 == 0:
			plot_list = []
			for iter in range(16):
				test = generated_image[iter].to(device)
				test = test.view(1,28,28)
				npimg = test.detach().numpy()
				plot_list.append(npimg)
				save_image(torch.tensor(plot_list), 'generated-images/'+str(epoch)+'-'+str(i)+'.png', nrow=4)

with torch.no_grad():
	test_list = []
	for i in range(16):
		i = i % 10
		condition_vector = torch.zeros(1,10).to(device)
		condition_vector[0][i] = 1
		test_image = torch.randn(1, args.latent_vector).to(device)
		test_image = generator(test_image, condition_vector).view(1,28,28).to(device)
		test_image = test_image.detach().numpy()
		test_list.append(test_image)
	save_image(torch.tensor(test_list), 'final-test-image.png', nrow=4)




