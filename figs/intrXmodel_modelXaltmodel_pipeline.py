"""
This script takes a number of models and first finds the best correlated
cell within each model for each interneuron recording and then for that
cell finds the best correlated cell in each other model within the grand
folder.
This results in a csv that has an entry for each
interneuron recording for each model for each model.

This script can be used by arguing experiment folders like so:

$ python3 intrXmodel_modelXaltmodel_pipeline.py <path_to_exp_folder1_here> <path_to_exp_folder2_here>

Each model from each of the experiment folders is first classified by
prep (i.e. dataset and stim_type) and then, one-by-one, each model
is correlated with each interneuron and then correlated with each of
its alternative seedings.

This script will save two csvs using the save_ext specified below.
One of the csvs is labeled "intr_" + save_ext. This one is used to
track interneuron correlations for tracking which units to find
correlations for. You will likely want to use only the main_df for
your purposes.
"""
import torch
import pandas as pd
import torchdeepretina as tdr
from torchdeepretina.utils import perm_similarity,mtx_cor
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import time
import sys
import os
import gc
import resource

DEVICE = torch.device("cuda:0")

if __name__=="__main__":
    n_samples = 5000 # Number of samples used for comparison
    verbose = True
    table_checkpts = True
    batch_size = 1000
    sim_folder = "csvs" # Folder to save comparison csv to
    save_ext = "cors.csv"
    intr_data_path = "~/interneuron_data"
    slide_steps = 0 # slides the intr stimulus to find better model correlations
    same_prep_only = False

    if not os.path.exists(sim_folder):
        os.mkdir(sim_folder)

    grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    pre_intr_df = None
    for i,grand_folder in enumerate(grand_folders):
        if ".csv" in grand_folder:
            pre_intr_df = grand_folder
            print("Using", pre_intr_df, "for interneuron correlations")
        else:
            print("Analyzing", grand_folder)
            paths = tdr.io.get_model_folders(grand_folder)
            paths = [os.path.join(grand_folder,path) for path in paths]
            if i == 0:
                model_paths = paths
            else:
                model_paths = model_paths + paths
    save_file = [ gfold.replace(".","") + "_" for gfold in grand_folders ]
    save_file = [sf.replace("training_scripts", "") for sf in save_file]
    save_file = [sf.replace("/","") for sf in save_file]
    save_file = "_".join(save_file) + save_ext
    save_file = os.path.join(sim_folder,save_file).replace("__", "_")
    print("Models:")
    print("\n".join(model_paths))
    print("Saving to:", save_file)

    if os.path.exists(save_file):
        main_df = pd.read_csv(save_file, sep="!")
    else:
        table = {
            "cell_file": [],
            "cell_idx": [],
            "cell_type": [],
            "intr_cor":[],

            "model1":[],
            "dataset1": [],
            "stim_type1": [],
            "seed1": [],
            "model2":[],
            "dataset2": [],
            "stim_type2": [],
            "seed2": [],
            "m1_layer":[],
            "chan1":[],
            "row1":[],
            "col1":[],
            "m2_layer":[],
            "chan2":[],
            "row2":[],
            "col2":[],
            "cor":[],
        }
        main_df = pd.DataFrame(table)
    intr_save_file = save_file.replace(save_ext, "intr_" + save_ext)
    main_intr_df = None
    if pre_intr_df is None:
        if os.path.exists(intr_save_file):
            pre_intr_df = pd.read_csv(intr_save_file, sep="!")
            print("Using", pre_intr_df, "for interneuron correlations")
    else:
        print("Using", pre_intr_df, "for interneuron correlations")
        pre_intr_df = pd.read_csv(pre_intr_df, sep="!")
        pre_intr_df["save_folder"] = "./" + pre_intr_df["save_folder"]
        pre_intr_df["save_folder"] = pre_intr_df.apply(
            lambda x: x.save_folder.split("/")[-1], axis=1
        )
        pre_intr_df = pre_intr_df.loc[pre_intr_df["intr_stim"]=="boxes"]

    # Load Both Regular and Interneuron Data
    if verbose:
        print("\nLoading Data")
    data = tdr.datas.loadexpt("15-10-07", "all", "naturalscene",
                                            'train', history=0)
    stim = data.X[:n_samples]
    interneuron_data = tdr.datas.load_interneuron_data(
                                            root_path=intr_data_path,
                                            filter_length=40,
                                            stim_keys={"boxes"},
                                            join_stims=True,
                                            window=True)

    stim_dict, mem_pot_dict, _ = interneuron_data
    for k in stim_dict.keys():
        stim_dict[k] =    {"test":stim_dict[k]}
        mem_pot_dict[k] = {"test":mem_pot_dict[k]}

    # Categorize Models into preps and store hyperparameters
    datasets = ["15-10-07", "15-11-21a", "15-11-21b"]
    stim_types = ["naturalscene", "whitenoise"]
    preps = {
        dset: {stype: [] for stype in stim_types} for dset in datasets
    }
    model_hyps = dict()
    for model_path in model_paths:
        hyps = tdr.io.get_hyps(model_path)
        model_hyps[model_path] = hyps
        if same_prep_only:
            dset = hyps["dataset"]
            stype = hyps["stim_type"]
            preps[dset][stype].append(model_path)
        else:
            for dset in datasets:
                for stype in stim_types:
                    preps[dset][stype].append(model_path)

    #Main For Loop For Finding Interneuron Cors and then Model-Model Cors
    responses = dict()
    for i,model_path in enumerate(model_paths):
        if verbose:
            print("\nBeginning model:", model_path,"| {}/{}".format(
                                            i,len(model_paths)))
            print()

        # Load Model
        model1 = tdr.io.load_model(model_paths[i])
        model1.eval()

        # Collect Interneuron Correlations
        with torch.no_grad():
            if verbose:
                print("Computing Intr Cors")
            model1.to(DEVICE)

            # Remove last layer because this is generally the ganglion layer
            m1_layers = tdr.utils.get_conv_layer_names(model1)[:-1]

            # Create new interneuron cors or use old ones
            intr_df = None
            path = ("./"+model_path).split("/")[-1]
            if pre_intr_df is not None and path in set(pre_intr_df["save_folder"]):
                intr_df = pre_intr_df.loc[pre_intr_df["save_folder"]==path]
                layers = set(intr_df["layer"])
                if len(m1_layers-layers) > 0:
                    intr_df = None
            if intr_df is None:
                intr_df = tdr.intracellular.get_intr_cors(model1, stim_dict,
                                               mem_pot_dict,
                                               layers=set(m1_layers),
                                               batch_size=batch_size,
                                               slide_steps=slide_steps,
                                               abs_val=True,
                                               verbose=verbose,
                                               window=False)

            # Keep only best correlations for each intr cell
            intr_df = intr_df.sort_values(by="cor", ascending=False)
            intr_df = intr_df.drop_duplicates(["cell_file", "cell_idx"])
            intr_df["save_folder"] = path
            if main_intr_df is None: main_intr_df = intr_df
            else: main_intr_df = main_intr_df.append(intr_df)

            if verbose:
                print("Computing Model1 Responses")
            if model_paths[i] in responses:
                m1_resp = responses[model_paths[i]]
            else:
                m1_resp, _ = tdr.analysis.get_resps(model1,
                                                stim,
                                                m1_layers,
                                                act=True,
                                                ig=False,
                                                batch_size=batch_size,
                                                to_numpy=True,
                                                verbose=verbose)
                responses[model_path] = m1_resp
            model1.cpu()

            # Collect indices that matter from model1 responses
            m1_indices = dict()
            if verbose:
                print("Determining maximal indices")
            for layer in m1_layers:
                shape = m1_resp[layer].shape
                temp = intr_df.loc[intr_df["layer"]==layer]
                print(layer, shape)
                assert len(shape)==4
                try:
                    chans = np.asarray(temp["chan"].astype("int"))
                    rows =  np.asarray(temp["row"].astype("int"))
                    cols =  np.asarray(temp["col"].astype("int"))
                    m1_indices[layer] = np.ravel_multi_index(
                        (chans, rows, cols),
                        shape[1:]
                    )
                except:
                    print("Chan")
                    for s in set(intr_df["chan"]):
                        print(s)
                    print("row")
                    for s in set(intr_df["row"]):
                        print(s)
                    print("col")
                    for s in set(intr_df["col"]):
                        print(s)

            # Loop over alternate models
            hyps = model_hyps[model_path]
            dset, stype = hyps["dataset"], hyps["stim_type"]
            stats_string = "\n"
            for j,comp_path in enumerate(preps[dset][stype]):
                # No Need to Look at Correlation with self
                if comp_path == model_path: continue

                model2 = tdr.io.load_model(comp_path)
                model2.eval()

                # Remove last layer because this is generally the ganglion layer
                m2_layers = tdr.utils.get_conv_layer_names(model2)[:-1]
                if comp_path in responses:
                    m2_resp = responses[comp_path]
                else:
                    model2.to(DEVICE)
                    m2_resp, _ = tdr.analysis.get_resps(model2,
                                                    stim,
                                                    m2_layers,
                                                    act=True,
                                                    ig=False,
                                                    batch_size=batch_size,
                                                    to_numpy=True,
                                                    verbose=verbose)
                    model2.cpu()
                    responses[comp_path] = m2_resp

                # Find correlations for important indices
                max_cors = dict()
                arg_maxes = dict()
                for l1 in m1_layers:
                    max_cors[l1] = dict()
                    arg_maxes[l1] = dict()
                    resp1 = m1_resp[l1]
                    shape1 = resp1.shape
                    resp1 = resp1.reshape(len(resp1),-1)
                    idxs = m1_indices[l1]
                    if len(idxs) == 0:
                        print("No Max Cors for Layer", l1)
                        continue
                    resp1 = resp1[:,idxs]

                    for l2 in m2_layers:
                        resp2 = m2_resp[l2]
                        shape2 = resp2.shape
                        resp2 = resp2.reshape(len(resp2),-1)

                        print("Resp1:", resp1.shape)
                        print("Resp2:", resp2.shape)
                        cors = mtx_cor(resp1,resp2,to_numpy=True)
                        maxs = np.max(cors,    axis=-1)
                        args = np.argmax(cors, axis=-1)

                        max_cors[l1][l2] =  maxs
                        arg_maxes[l1][l2] = args

                # Record appropriate correlations
                for l1 in m1_layers:
                    df = intr_df.loc[intr_df["layer"]==l1]
                    if len(df) == 0:
                        print("No Max Cors for Layer", l1)
                        continue
                    print("df len:", len(df))
                    chans = np.asarray(df["chan"])
                    rows =  np.asarray(df["row"])
                    cols =  np.asarray(df["col"])
                    shape = m1_resp[l1].shape
                    idxs = np.ravel_multi_index(
                        (chans,rows,cols), shape[1:]
                    ).squeeze()

                    for l2 in max_cors[l1].keys():
                        for r in range(len(df)):
                            data = intr_df.iloc[r]
                            table["cell_file"].append(data["cell_file"])
                            table["cell_idx"].append(data["cell_idx"])
                            table["cell_type"].append(data["cell_type"])
                            table["intr_cor"].append(data["cor"])

                            table['model1'].append(model_path)
                            hyps1 = model_hyps[model_path]
                            table['dataset1'].append(hyps1["dataset"])
                            table['stim_type1'].append(hyps1["stim_type"])
                            table['seed1'].append(hyps1["seed"])

                            table['model2'].append(comp_path)
                            hyps2 = model_hyps[comp_path]
                            table['dataset2'].append(hyps2["dataset"])
                            table['stim_type2'].append(hyps2["stim_type"])
                            table['seed2'].append(hyps2["seed"])

                            table['m1_layer'].append(l1)
                            table['chan1'].append(chans[r])
                            table["row1"].append(rows[r])
                            table['col1'].append(cols[r])

                            table['m2_layer'].append(l2)
                            shape = m2_resp[l2].shape
                            val = arg_maxes[l1][l2][r]
                            chan,row,col=np.unravel_index(val,shape[1:])
                            table['chan2'].append(chan)
                            table["row2"].append(row)
                            table['col2'].append(col)
                            table['cor'].append( max_cors[l1][l2][r] )

                    for l2 in max_cors[layer].keys():
                        stats_string += "{}-{}: {}\n".format(layer, l2,
                                          np.mean(max_cors[layer][l2]))

            if verbose:
                print(stats_string)

            df = pd.DataFrame(table)
            table = {k:[] for k in table.keys()}
            if len(main_df['cor']) == 0:
                main_df = df
            else:
                main_df = main_df.append(df, sort=True)
            main_df.to_csv(save_file, sep="!",
                             header=True, index=False)
            main_intr_df.to_csv(intr_save_file, sep="!",
                             header=True, index=False)
            gc.collect()
            max_mem_used = resource.getrusage(resource.RUSAGE_SELF)
            max_mem_used = max_mem_used.ru_maxrss/1024
            print("Memory Used: {:.2f} mb".format(max_mem_used))
            gpu_mem = tdr.utils.get_gpu_mem()
            s = ["gpu{}: {}".format(k,v) for k,v in gpu_mem.items()]
            s = "\n".join(s)
            print(s)
            print()
