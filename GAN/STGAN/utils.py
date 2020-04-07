import torch
from torch import nn

# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid(data):
	return torch.ones(data.size()).to(device)

def wrong(data):
	return torch.zeros(data.size()).to(device)


