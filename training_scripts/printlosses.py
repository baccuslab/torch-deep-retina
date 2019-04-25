folder = 'recreate'
exp_id = 'recreate_0'
for epoch in range(100):
	check = io.load_checkpoint(exp_id, epoch, folder, return_ckpt=True)
	print('Loss: ' + check['loss'] + ',  Optimizer: ' + check['optimizer_state_dict'])
