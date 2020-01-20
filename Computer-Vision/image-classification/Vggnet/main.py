import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


"Re-implementation of VGGnet-16 with Cifar-10. The kernel sizes and learning rate are slightly different from the original model."

learning_rate = 0.0001
num_epochs = 5
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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

class VGGnet(nn.Module):

  def __init__(self, num_classes = 10):
    super(VGGnet, self).__init__()
    self.features = nn.Sequential(
      # shape: [3,32,32]
      nn.Conv2d(3, 64, kernel_size=3, padding = 1),
      nn.ReLU(inplace = True), 
      # shape: [64,32,32]
      nn.Conv2d(64, 64, kernel_size=3, padding = 1),
      nn.ReLU(inplace = True), 
      # shape: [64,32,32]
      nn.MaxPool2d(kernel_size=2, stride = 2),
      # shape: [64,16,16]
      nn.Conv2d(64, 128, kernel_size=3, padding = 1),
      nn.ReLU(inplace = True), 
      # shape: [128,16,16]
      nn.Conv2d(128, 128, kernel_size=3, padding = 1),
      nn.ReLU(inplace = True), 
      # shape: [128,16,16]
      nn.MaxPool2d(kernel_size=2, stride = 2),
      # shape: [128,8,8]
      nn.Conv2d(128, 256, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [256,8,8]
      nn.Conv2d(256, 256, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [256,8,8]
      nn.Conv2d(256, 256, kernel_size=1, padding =1),
      nn.ReLU(inplace = True),
      # shape: [256,10,10]
      nn.MaxPool2d(kernel_size=2, stride = 2),
      # shape: [256,5,5]
      nn.Conv2d(256, 512, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,5,5]
      nn.Conv2d(512, 512, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,5,5]
      nn.Conv2d(512, 512, kernel_size=1, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,7,7]
      nn.Conv2d(512, 512, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,7,7]
      nn.Conv2d(512, 512, kernel_size=3, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,7,7]
      nn.Conv2d(512, 512, kernel_size=1, padding =1),
      nn.ReLU(inplace = True),
      # shape: [512,9,9]
      nn.MaxPool2d(kernel_size=2, stride = 1),
      # shape: [512,8,8]
      
    ) 
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(512*8*8, 4096),
      nn.ReLU(inplace = True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace = True),
      nn.Linear(4096, num_classes),
      nn.Softmax(dim = 1),      
      )
  def forward(self, x):
    x = self.features(x); 
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x 


if __name__ == "__main__":
  # test()
  model = VGGnet(len(classes)).to(device)
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
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
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


