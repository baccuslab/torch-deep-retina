from scipy.stats import pearsonr, zscore
import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import h5py as h5
import os.path as path
from torch.distributions import normal
import gc
import resource
sys.path.append('../')
sys.path.append('../utils/')
from utils.miscellaneous import ShuffledDataSplit
from models import BNCNN, AbsBNBNCNN
import retio as io
import argparse
import time
from tqdm import tqdm
import json
import math
from utils.deepretina_loader import loadexpt
from utils.stimuli import concat
from utils.physiology import Physio
from pyret.filtertools import filterpeak
from utils.intracellular import pad_to_edge, classify

DEVICE = torch.device("cuda:0")

seed = 3
np.random.seed(seed)
torch.manual_seed(seed)


def prep_dataGang(dataGang, batch_size):
	train_data = dataGang[0]
	test_data = dataGang[1]
	num_val = 20000
	data = ShuffledDataSplit(train_data, num_val)

	data.torch()
	epoch_length = data.train_shape[0]
	num_batches,leftover = divmod(epoch_length, batch_size)
	print("Train size:", epoch_length)
	print("Val size:", data.val_shape[0])
	print("N Batches:", num_batches, "  Leftover:", leftover)
	return train_data, test_data, data, num_batches

def prep_dataIntra(dataIntra, num_batches):
	intraData = DataContainer(concat(pad_to_edge(zscore(dataIntra[0]))), dataIntra[1], mode=False)
	intra_batchsize, leftover = divmod(intraData.X.shape[0], num_batches)
	intraData = ShuffledDataSplit(intraData, 150, intra_batchsize)
	intraData.torch()
	return intraData, intra_batchsize

def load_pretrained(model, pretrained_path):
	pretrained_dict = torch.load(pretrained_path)['model_state_dict']
	model_dict = model.state_dict()
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	return model

def train(hyps, model, dataGang, dataIntra, intraRF):

	def batch_loop(batch, model, optimizer):
		def prepare_x_and_label(data, idxs):
			x = data.train_X[idxs]
			x = x.to(DEVICE)

			label = data.train_y[idxs]
			label = label.float()
			label = label.to(DEVICE)
			return x, label
		optimizer.zero_grad()
		idxs = indices[batch_size*batch:batch_size*(batch+1)]
		intraidxs = intraindices[intra_batchsize*batch:intra_batchsize*(batch+1)]

		x, label = prepare_x_and_label(data, idxs)
		y = model(x.to(DEVICE))
		y = y.float() 
		activity_l1 = LAMBDA1 * torch.norm(y, 1).float()/y.shape[0]
		error_gang = loss_fn(y,label) + activity_l1

		del x
		del label

		intra_x, intra_label = prepare_x_and_label(intraData, intraidxs)
		intra_y = physio.inspect(intra_x, insp_keys={layer})[layer][:, celltype, sidx[0], sidx[1]]
		error_intra = loss_fn(intra_y, intra_label)

		if gradnorm:
			if loss_intra0 is None:
				loss_intra0 = error_intra
				loss_gang0 = error_gang

			loss = loss_weights[0]*error_gang + loss_weights[1]*error_intra
			loss_gradients = []
		error_gang.backward()
		error_intra.backward()

		if gradnorm:
			loss_gradients.append(torch.norm(loss_weights[0]*torch.sum(model.sequential[11].weight), 1))
			loss_gradients.append(torch.norm(loss_weights[1]*torch.sum(model.sequential[6].weight), 1))
			total_loss_gradient = (loss_weights[0]*loss_gradients[0] + loss_weights[1]*loss_gradients[1])/T
			loss_ratios = []
			loss_ratios.append(error_gang/loss_gang0)
			loss_ratios.append(error_intra/loss_intra0)
			total_loss_ratio = (loss_weights[0]*loss_ratios[0] + loss_weights[1]*loss_ratios[1])/T
			inverse_rates = []
			inverse_rates.append(loss_ratios[0]/total_loss_ratio)
			inverse_rates.append(loss_ratios[1]/total_loss_ratio)

			L_grad = torch.zeros(1, requires_grad = True)
			for i in range T:
				constant = total_loss_gradient*np.power(inverse_rates[i], alpha)
				constant.requires_grad = False
				L_grad += torch.norm(loss_gradients[i] - constant)

			L_grad.backward()

		loss = error_intra + error_gang
		optimizer.step()
		if gradnorm:
			normalize_coeff = T/ torch.sum(loss_weights)
        	loss_weights = loss_weights * normalize_coeff
		print("Loss:", loss.item()," - error_gang:", error_gang.item(), "- error_intra:", error_intra.item(), " - l1:", activity_l1.item(), " | ", int(round(batch/num_batches, 2)*100), "% done", end='               \r')
		if math.isnan(epoch_loss) or math.isinf(epoch_loss):
			return None

		del y
		del intra_x
		del intra_label
		del intra_y
		return error_gang.item(), error_intra.item()

	def validate_model(model, scheduler):
		model.eval()
		val_preds = []
		val_loss = 0
		step_size = 2500
		n_loops = data.val_shape[0]//step_size
		for v in tqdm(range(0, n_loops*step_size, step_size)):
			temp = model(data.val_X[v:v+step_size].to(DEVICE)).detach()
			val_loss += loss_fn(temp, data.val_y[v:v+step_size].to(DEVICE)).item()
			val_preds.append(temp.cpu().numpy())
		val_loss = val_loss/n_loops
		val_preds = np.concatenate(val_preds, axis=0)
		val_acc = np.mean([pearsonr(val_preds[:, i], data.val_y[:val_preds.shape[0]][:,i].numpy()) for i in range(val_preds.shape[-1])])
		print("Val Acc:", val_acc, " -- Val Loss:", val_loss, " | SaveFolder:", SAVE)
		scheduler.step(val_loss)

		test_obs = model(test_x.to(DEVICE)).cpu().detach().numpy()

		avg_pearson = 0
		for cell in range(test_obs.shape[-1]):
			obs = test_obs[:,cell]
			lab = test_data.y[:,cell]
			r,p = pearsonr(obs,lab)
			avg_pearson += r
			print('Cell ' + str(cell) + ': ')
			print('-----> pearsonr: ' + str(r))
		avg_pearson = avg_pearson / float(test_obs.shape[-1])
		physio.remove_hooks()
		physio.remove_refs()
		save_dict = {
			"model": model,
			"loss": avg_loss,
			"loss_intra": avg_loss_intra,
			"loss_gang": avg_loss_gang,
			"epoch":epoch,
			"val_loss":val_loss,
			"val_acc":val_acc,
			"test_pearson":avg_pearson,
			"loss_weights":loss_weights,
		}
		io.save_checkpoint_dict(save_dict,SAVE,'test')
		del val_preds
		del temp
		print()
		# If loss is nan, training is futile
		if math.isnan(avg_loss) or math.isinf(avg_loss):
			return -1
		return val_loss, val_acc, avg_pearson

	"""%%%%%%%%%%%%%%%%%%%%%%% Actual start of train %%%%%%%%%%%%%%%%%%%%%"""



	


	LR = hyps['lr']
	LAMBDA1 = hyps['l1']
	LAMBDA2 = hyps['l2']
	EPOCHS = hyps['n_epochs']
	batch_size = hyps['batch_size']
	gradnorm = hyps['gradnorm']

	SAVE = hyps['save_folder']
	if not os.path.exists(SAVE):
		os.mkdir(SAVE)
	with open(SAVE + "/hyperparams.txt",'w') as f:
		f.write(str(model)+'\n')
		for k in sorted(hyps.keys()):
			f.write(str(k) + ": " + str(hyps[k]) + "\n")

	train_data, test_data, data, num_batches = prep_dataGang(dataGang, batch_size)
	intraData, intra_batchsize = prep_dataIntra(dataIntra, num_batches)

	print(model)
	model = model.to(DEVICE)
	physio = Physio(model, numpy=True)

	if 'pretrained_path' in hyps:
		model = load_pretrained(model, hyps['pretrained_path'])
		stim = torch.from_numpy(concat(pad_to_edge(zscore(dataIntra[0])[:5000]))).to(DEVICE)
		model_response = physio.inspect(stim, insp_keys={"sequential.0", "sequential.6"})
		del stim
		layer, celltype, sidx, _ = classify(dataIntra[1], model_response, 4960, layer_keys=["sequential.0", "sequential.6"])
		del model_response
	else:
		_, sidx, tidx = filterpeak(intraRF)
		layer = "sequential.6"
		celltype = 0

	physio.numpy = False
	torch.cuda.empty_cache()

	loss_fn = torch.nn.PoissonNLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = LAMBDA2)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.2*LR)

	# train/val split

	# test data
	test_x = torch.from_numpy(test_data.X)
	T = 2
	loss_weights = torch.ones(T, requires_grad=True)
	loss_intra0 = None
	loss_gang0 = None

	for epoch in range(EPOCHS):
		model.train(mode=True)
		indices = torch.randperm(data.train_shape[0]).long()
		intraindices = torch.randperm(intraData.train_shape[0]).long()

		losses = []
		epoch_loss_gang = 0
		epoch_loss_intra = 0
		epoch_loss = 0
		print('Epoch ' + str(epoch))  
		
		starttime = time.time()

		for batch in range(num_batches):
			losses = batch_loop(batch, model, optimizer)
			if losses == None: 
				break
			epoch_loss_gang += losses[0]
			epoch_loss_intra += losses[1]
			epoch_loss = epoch_loss_gang + epoch_loss_intra

		avg_loss_gang = epoch_loss_gang/num_batches
		avg_loss_intra = epoch_loss_intra/num_batches
		avg_loss = avg_loss_gang + avg_loss_intra
		print('\nAvg Loss: ' + str(avg_loss), " - exec time:", time.time() - starttime)

		#validate model
		ret = validate_model(model, scheduler)
		if ret == -1: break


	if ret != -1:
		val_acc, val_loss, avg_pearson = ret
		results = {"Loss":avg_loss, "ValAcc":val_acc, "ValLoss":val_loss, "TestPearson":avg_pearson}
		with open(SAVE + "/hyperparams.txt",'a') as f:
			f.write("\n" + " ".join([k+":"+str(results[k]) for k in sorted(results.keys())]) + '\n')
	return results

def hyper_search(hyps, hyp_ranges, keys, train, idx=0):
	"""
	Recursive function to loop through each of the hyperparameter combinations

	hyps - dict of hyperparameters created by a HyperParameters object
		type: dict
		keys: name of hyperparameter
		values: value of hyperparameter
	hyp_ranges - dict of ranges for hyperparameters to take over the search
		type: dict
		keys: name of hyperparameters to be searched over
		values: list of values to search over for that hyperparameter
	keys - keys of the hyperparameters to be searched over. Used to
			specify order of keys to search
	train - method that handles training of model. Should return a dict of results.
	idx - the index of the current key to be searched over
	"""
	# Base call, runs the training and saves the result
	if idx >= len(keys):
		if 'exp_num' not in hyps:
			if 'starting_exp_num' not in hyps: hyps['starting_exp_num'] = 0
			hyps['exp_num'] = hyps['starting_exp_num']
			if not os.path.exists(hyps['exp_name']):
				os.mkdir(hyps['exp_name'])
			hyps['results_file'] = hyps['exp_name']+"/results.txt"
		hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
		for k in keys:
			hyps['save_folder'] += "_" + str(k)+str(hyps[k])
		print("Loading", hyps['stim_type'],"using Cells:", hyps['cells'], "from dataset:", hyps['dataset'])
		train_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],
											hyps['stim_type'],'train',40,0))
		norm_stats = [train_data.stats['mean'], train_data.stats['std']] 
		test_data = DataContainer(loadexpt(hyps['dataset'],hyps['cells'],hyps['stim_type'],
														'dev',40,0, norm_stats=norm_stats))
		test_data.X = test_data.X[:500]
		test_data.y = test_data.y[:500]

		file = '/home/julia/julia/data/amacrines_late_2012.h5'
		with h5.File(file, 'r') as f:
			intraData = (np.array(f['boxes/stimuli']), np.array(f['boxes/detrended_membrane_potential'][0, 40:]))
			intraRF = np.array(f['boxes/rfs/040412_c1'])
		data = [train_data, test_data]
		if "chans" in hyps:
			model = hyps['model_type'](test_data.y.shape[-1], noise=hyps['noise'], bias=hyps['bias'], chans=hyps['chans'])
		else:
			model = hyps['model_type'](test_data.y.shape[-1], noise=hyps['noise'], bias=hyps['bias'])
		results = train(hyps, model, data, intraData, intraRF)
		with open(hyps['results_file'],'a') as f:
			if hyps['exp_num'] == hyps['starting_exp_num']:
				f.write(str(model)+'\n\n')
				f.write("Hyperparameters:\n")
				for k in hyps.keys():
					if k not in hyp_ranges:
						f.write(str(k) + ": " + str(hyps[k]) + '\n')
				f.write("\nHyperranges:\n")
				for k in hyp_ranges.keys():
					f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
				f.write('\n')
			results = " ".join([k+":"+str(results[k]) for k in sorted(results.keys())])
			f.write(hyps['save_folder'].split("/")[-1] + ":\n\t" + results +"\n\n")
		hyps['exp_num'] += 1

	# Non-base call. Sets a hyperparameter to a new search value and passes down the dict.
	else:
		key = keys[idx]
		for param in hyp_ranges[key]:
			hyps[key] = param
			hyper_search(hyps, hyp_ranges, keys, train, idx+1)
	return

def set_model_type(model_str):
	if model_str == "BNCNN":
		return BNCNN
	if model_str == "AbsBNBNCNN":
		return AbsBNBNCNN
	print("Invalid model type!")
	return None

def load_json(file_name):        

	with open(file_name) as f:
		s = f.read()
		j = json.loads(s)
	return j

class DataContainer():
	def __init__(self, data, datay=None, mode = True):
		if mode:
			self.X = data.X
			self.y = data.y
			self.stats = data.stats
		else:
			self.X = data
			self.y = datay
			self.stats = None

if __name__ == "__main__":
	hyperparams_file = "hyperparams.json"
	hyperranges_file = 'hyperranges.json'
	hyps = load_json(hyperparams_file)
	inp = input("Last chance to change the experiment name "+hyps['exp_name']+": ")
	inp = inp.strip()
	if inp is not None and inp != "":
		hyps['exp_name'] = inp
	hyp_ranges = load_json(hyperranges_file)
	print("Model type:", hyps['model_type'])
	hyps['model_type'] = set_model_type(hyps['model_type'])
	keys = list(hyp_ranges.keys())
	print("Searching over:", keys)

	hyper_search(hyps, hyp_ranges, keys, train, 0)




