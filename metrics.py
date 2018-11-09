import torch

def correlation_coefficient(obs_rate, est_rate):
	"""Pearson correlation coefficient"""
	x_mu = obs_rate - torch.mean(obs_rate, dim = 0, keepdim = True)
	x_std = torch.std(obs_rate, dim = 0, keepdim= True)
	y_mu = est_rate - torch.mean(est_rate, dim = 0, keepdim = True)
	y_std = torch.std(est_rate, dim = 0, keepdim = True)

	return torch.mean(x_mu* y_mu, dim = 0, keepdim = True) / (x_std * y_std)

def mean_squared_error(obs_rate, est_rate):
	return torch.mean(torch.square(est_rate - obs_rate), dim = 0, keepdim = True)

def fraction_of_explained_variance(obs_rate, est_rate):
	return 1.0 - mean_squared_error(obs_rate, est_rate)/ torch.var(obs_rate, dim=0, keepdims= True)



