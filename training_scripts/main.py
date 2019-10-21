"""
Used as a general training script. Can give command line arguments to specify which json files to use
for hyperparameters and the ranges that those hyperparameters can take on.

$ python3 general_training.py params=hyperparams.json ranges=hyperranges.json

Defaults to hyperparams.json and hyperranges.json if no arguments are provided
"""
import sys
import os
import time
import numpy as np
import torch
import select
from torchdeepretina.training import hyper_search
from torchdeepretina.utils import load_json
from torchdeepretina.analysis import analysis_pipeline

if __name__ == "__main__":
    hyperparams_file = "hyps/hyperparams.json"
    hyperranges_file = 'hyps/hyperranges.json'
    device = 0
    if len(sys.argv) > 1:
        for i,arg in enumerate(sys.argv[1:]):
            temp = sys.argv[1].split("=")
            if len(temp) > 1:
                if "params" == temp[0]:
                    hyperparams_file = temp[1]
                elif "ranges" == temp[0]:
                    hyperranges_file = temp[1]
            else:
                if i == 0:
                    hyperparams_file = arg
                elif i == 1:
                    hyperranges_file = arg
    print()
    print("Using hyperparams file:", hyperparams_file)
    print("Using hyperranges file:", hyperranges_file)
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

    if "shift_labels" in hyps and hyps['shift_labels']:
        print("{0} WARNING: YOU ARE USING SHIFTED LABELS {0}".format("!"*5))

    sleep_time = 8
    if os.path.exists(hyps['exp_name']):
        _, subds, _ = next(os.walk(hyps['exp_name']))
        nums = []
        for d in subds:
            try:
                nums.append(int(d.split("_")[1]))
            except:
                pass
        nums = sorted(nums)
        if len(nums) == 0 and hyps['starting_exp_num'] is None:
            hyps['starting_exp_num'] = 0
        elif hyps['starting_exp_num'] is None:
            hyps['starting_exp_num'] = nums[-1] + 1
        if int(hyps['starting_exp_num']) in nums:
            print("{0} WARNING: YOU ARE CURRENTLY DUPLICATING EXP NUMS {0}".format("!"*5))
            print("Would you like to use", nums[-1]+1, "instead? (Y/n)")
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
            if not i or (i and sys.stdin.readline().strip().lower() == "y"):
                hyps['starting_exp_num'] = nums[-1]+1
            elif i and sys.stdin.readline().strip().lower() != "n" and sys.stdin.readline().strip().lower() != "":
                try:
                    hyps['starting_exp_num'] = int(sys.stdin.readline().strip().lower())
                except:
                    print("Error parsing input")
                    pass
        else:
            print("You have "+str(sleep_time)+" seconds to cancel experiment name "+
                        hyps['exp_name']+" (num "+ str(hyps['starting_exp_num'])+"): ")
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
            if i:
                try:
                    hyps['starting_exp_num'] = int(sys.stdin.readline().strip().lower())
                except:
                    pass
    else:
        if hyps['starting_exp_num'] is None:
            hyps['starting_exp_num'] = 0
        print("You have "+str(sleep_time)+" seconds to cancel experiment name "+
                    hyps['exp_name']+" (num "+ str(hyps['starting_exp_num'])+"): ")
        i,_,_ = select.select([sys.stdin], [],[],sleep_time)
        if i:
            try:
                hyps['starting_exp_num'] = int(sys.stdin.readline().strip().lower())
            except:
                pass
    print("Using start num:", hyps['starting_exp_num'])
    print()

    keys = list(hyp_ranges.keys())
    start_time = time.time()
    hyper_search(hyps, hyp_ranges, keys, device)
    print("Total Execution Time:", time.time() - start_time)
    print("\n\nBeginning Analysis..")
    exp_folder = hyps['exp_name']
    dfs = analysis_pipeline(exp_folder, make_figs=True, verbose=True)
    for k in dfs.keys():
        dfs[k].to_csv(os.path.join(exp_folder,k),sep="!",header=True,index=False)

