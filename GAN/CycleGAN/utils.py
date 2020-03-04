import torch
from torch import nn

# file revised 	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid(data):
	return torch.ones(1, data.shape[1], data.shape[2], data.shape[3]).to(device)

def wrong(data):
	return torch.zeros(1, data.shape[1], data.shape[2], data.shape[3]).to(device)


