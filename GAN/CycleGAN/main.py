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
from data import *
from model import *
from utils import *
# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting args
parser = argparse.ArgumentParser(description = "Cycle GAN")
parser.add_argument('--batch_size', type = int, default =1, help = "batch_size")
parser.add_argument('--epoch', type = int, default = 200, help = "epoch")
parser.add_argument('--d_learning_rate', type = float, default = 0.00001, help = "learning_rate")
parser.add_argument('--g_learning_rate', type = float, default = 0.00001, help = "learning_rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_vector", type=int, default=100, help="latent dimension")
parser.add_argument("--recon_lambda", type= float, default=5.0, help="identity lambda")
parser.add_argument("--cycle_lambda", type=float, default=10.0, help="cycle lambda")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks")
parser.add_argument("--out_nc", type=int, default=64, help="number of output channels")
args = parser.parse_args()

os.makedirs('generated-images', exist_ok = True)


transform_ = transforms.Compose([
	transforms.Resize(256),
	transforms.ToTensor(),
	transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])



train_data = PairedDataset('dataset/vangogh2photo', train = True, transform = transform_)
test_data = PairedDataset('dataset/vangogh2photo', train = False, transform = transform_)

train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True)


generator_A2B = Generator(args.out_nc, args.n_residual_blocks).to(device)
generator_B2A = Generator(args.out_nc, args.n_residual_blocks).to(device)
discriminator_A = Discriminator(args.out_nc).to(device)
discriminator_B = Discriminator(args.out_nc).to(device)

g_optimizer = optim.Adam(itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()), lr = args.g_learning_rate, betas = (args.b1, args.b2))
d_A_optimizer = optim.Adam(discriminator_A.parameters(), lr = args.d_learning_rate, betas = (args.b1, args.b2))
d_B_optimizer = optim.Adam(discriminator_B.parameters(), lr = args.d_learning_rate, betas = (args.b1, args.b2))

criterion_gan = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

generator_A2B.train()
generator_B2A.train()
discriminator_A.train()
discriminator_B.train()



for epoch in range(args.epoch):
	for i, (A, B) in enumerate(train_loader):

		real_A = A.float().to(device)
		real_B = B.float().to(device)

		g_optimizer.zero_grad()

		# identity_loss
		post_A = generator_B2A(real_A)
		identity_A_loss = criterion_identity(post_A, real_A)
		post_B = generator_A2B(real_B)
		identity_B_loss = criterion_identity(post_B, real_B)
		
		identity_loss = (identity_A_loss + identity_B_loss) / 2		

		# generator_loss
		fake_B = generator_A2B(real_A).to(device)
		gan_B_loss = criterion_gan(discriminator_B(fake_B), valid(discriminator_B(fake_B)))
		fake_A = generator_B2A(real_B).to(device)
		gan_A_loss = criterion_gan(discriminator_A(fake_A), valid(discriminator_A(fake_A)))

		gan_loss = (gan_A_loss + gan_B_loss) / 2

		# cycle_loss
		recon_A = generator_B2A(fake_B)	
		cycle_A_loss = criterion_cycle(recon_A, real_A)
		recon_B = generator_A2B(fake_A)	
		cycle_B_loss = criterion_cycle(recon_B, real_B)
		
		cycle_loss = (cycle_A_loss + cycle_B_loss) / 2

		g_loss = args.recon_lambda * identity_loss + gan_loss + args.cycle_lambda *cycle_loss

		g_loss.backward(retain_graph = True)
		g_optimizer.step()


		# real_loss
		# train discriminator		
		d_A_optimizer.zero_grad()
		disc_real_A = discriminator_A(real_A)
		d_A_loss_real = criterion_gan(disc_real_A, valid(disc_real_A))
		d_A_loss_fake = criterion_gan(discriminator_A(fake_A), wrong(discriminator_A(fake_A)))
		d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
		d_A_loss.backward()
		d_A_optimizer.step()
		
		d_B_optimizer.zero_grad()
		disc_real_B = discriminator_B(real_B)
		d_B_loss_real = criterion_gan(disc_real_B, valid(disc_real_B))
		d_B_loss_fake = criterion_gan(discriminator_B(fake_B), wrong(discriminator_B(fake_B)))
		d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
		d_B_loss.backward()
		d_B_optimizer.step()

		
		d_loss = d_A_loss + d_B_loss

		total_loss = d_loss + g_loss
		if i % 20 == 0:	
			print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, G-Loss: {:.4f}, D-Loss: {:.4f} '.format(epoch+1, args.epoch, i, int(len(train_loader)), total_loss.item(), g_loss, d_loss))
			plot_list = []
			plot_list.append(real_A[0])
			plot_list.append(fake_B[0])
			plot_list.append(recon_A[0])
			plot_list.append(real_B[0])
			plot_list.append(fake_A[0])
			plot_list.append(recon_B[0])
			plot_list = torch.stack(plot_list)
			save_image(plot_list, 'generated-images/'+str(epoch+1)+'_'+str(i)+'.png', nrow=6, normalize = True)

