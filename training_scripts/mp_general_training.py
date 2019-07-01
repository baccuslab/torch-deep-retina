"""
Used as a general training script. Can give command line arguments to specify which json files to use
for hyperparameters and the ranges that those hyperparameters can take on.

$ python3 general_training.py params=hyperparams.json ranges=hyperranges.json devices=0,1 cuda_buffer=3000 ram_buffer=6000 n_workers=4

Defaults to hyperparams.json and hyperranges.json if no arguments are provided
"""
import sys
import time
import numpy as np
import torch
from torchdeepretina.training import mp_hyper_search
from torchdeepretina.utils import load_json
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    hyperparams_file = "hyps/hyperparams.json"
    hyperranges_file = 'hyps/hyperranges.json'
    visible_devices = {0,1,2,3}
    cuda_buffer = 3000
    ram_buffer = 6000
    n_workers = 4
    if len(sys.argv) > 1:
        for i,arg in enumerate(sys.argv[1:]):
            temp = sys.argv[1].split("=")
            if len(temp) > 1:
                if "params" == temp[0]:
                    hyperparams_file = temp[1]
                elif "ranges" == temp[0]:
                    hyperranges_file = temp[1]
                elif "devices" == temp[0]:
                    visible_devices = set([int(x) for x in temp[1].split(",")])
                elif "cuda_buffer" == temp[0]:
                    cuda_buffer = int(temp[1])
                elif "ram_buffer" == temp[0]:
                    ram_buffer = int(temp[1])
                elif "n_workers" == temp[0]:
                    n_workers = int(temp[1])
            else:
                if i == 0 or "hyperparams" in arg:
                    hyperparams_file = arg
                elif i == 1 or "hyperranges" in arg:
                    hyperranges_file = arg
    print()
    print("Using hyperparams file:", hyperparams_file)
    print("Using hyperranges file:", hyperranges_file)
    print("Visible Devices:", visible_devices)
    print("CUDA Buffer:", cuda_buffer)
    print("RAM Buffer:", ram_buffer)
    print("N Workers:", n_workers)
    print()

    hyps = load_json(hyperparams_file)
    hyp_ranges = load_json(hyperranges_file)
    hyps_str = ""
    for k,v in hyps.items():
        if k not in hyp_ranges:
            hyps_str += "{}: {}\n".format(k,v)
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k,v) for k,v in hyp_ranges.items()]))

    # Random Seeds
    seed = 3
    if "rand_seed" in hyps:
        seed = hyps['rand_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)


    #sleep_time = 8
    #print("You have "+str(sleep_time)+" seconds to cancel experiment name "+
    #            hyps['exp_name']+" (num "+ str(hyps['starting_exp_num'])+"): ")
    #time.sleep(sleep_time)

    keys = list(hyp_ranges.keys())
    start_time = time.time()
    mp_hyper_search(hyps, hyp_ranges, keys, visible_devices=visible_devices, 
                          n_workers=n_workers, cuda_buffer=cuda_buffer, ram_buffer=ram_buffer)
    print("Total Execution Time:", time.time() - start_time)

