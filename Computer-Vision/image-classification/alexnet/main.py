import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


"Re-implementation of Alexnet with Cifar-10. The kernel sizes and learning rate are slightly different from the original model. 64.8% accuracy of test image with 15 epochs"

learning_rate = 0.0001
num_epochs = 15
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# transform = transforms.Compose(
#     [transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True,
										download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=False,
									   download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
										  shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
										 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

def test():
  dataiter = iter(train_loader)
  images, labels = dataiter.next()
  imshow(torchvision.utils.make_grid(images))
  print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Alexnet(nn.Module):

	def __init__(self, num_classes = 10):
		super(Alexnet, self).__init__()
		self.features = nn.Sequential(
		# shape: [3,32,32]
		nn.Conv2d(3, 96, kernel_size=4, stride = 4, padding = 2),
		nn.ReLU(inplace = True), 
		# shape: [96,9,9]
		nn.MaxPool2d(kernel_size=3, stride = 2),
		# shape: [96,4,4]
		nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
		nn.ReLU(inplace = True),
		# shape: [256,4,4]
		nn.MaxPool2d(kernel_size=2),
		# shape: [256,3,3]
		nn.Conv2d(256, 384, kernel_size=3, padding =1),
		nn.ReLU(inplace = True),
		# shape: [384,2,2]
		nn.Conv2d(384, 384, kernel_size=3, padding =1),
		nn.ReLU(inplace = True),
		# shape: [384,2,2]
		nn.Conv2d(384, 256, kernel_size=3, padding =1),
		nn.ReLU(inplace = True),
		# shape: [256,2,2]
		nn.MaxPool2d(kernel_size=2),
		# shape: [256,1,1]
		) 
		# self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256*1*1, 4096),
			nn.ReLU(inplace = True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace = True),
			nn.Linear(4096, num_classes),			
			)
	def forward(self, x):
		x = self.features(x); 
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x 


if __name__ == "__main__":
  # test()
	model = Alexnet(len(classes)).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
	total_step = len(train_loader)
	summary(model, (3, 32, 32))
	for epoch in range(num_epochs):
		for i, (images,labels) in enumerate(train_loader):
			images = images.to(device) 
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % 50 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.to(device) 
			labels = labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (labels == predicted).sum().item()
			print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
	"Maximum Accuray: 63.4% with learning_rate = 0.0001 & num_epochs = 15"
	 



