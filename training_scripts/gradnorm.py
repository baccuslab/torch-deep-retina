def gradnorm(alpha):
	T = 2
	loss_weights = torch.ones(T, requires_grad=True)
	loss_intra0 = None
	loss_gang0 = None
	for batch in range(num_batches):
		if loss_intra0 is None:
			loss_intra0 = error_intra
			loss_gang0 = error_gang

		loss = loss_weights[0]*error_gang + loss_weights[1]*error_intra
		loss_gradients = []
		error_gang.backward()
		error_intra.backward()

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
		optimizer.step()
		#normalize loss_weights
		normalize_coeff = T/ torch.sum(loss_weights)
        loss_weights = loss_weights * normalize_coeff



		
