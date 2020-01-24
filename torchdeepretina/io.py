import torch
import pickle
from torchdeepretina.models import *
from torchdeepretina.custom_modules import *
import torchdeepretina.utils as tdrutils
import os

def save_ln(model,file_name,hyps=None):
    """
    Saves an LNModel to file

    model: LNModel object (see models.py)
    file_name: str
        path to save model to
    """
    model_dict = {
                  "filt":  model.filt, 
                  "fit":   model.poly.coefficients,
                  "span":  model.span,
                  "center":model.center,
                  "norm_stats":model.norm_stats,
                  "cell_file":model.cell_file,
                  "cell_idx": model.cell_idx,
    }
    if hyps is not None:
        model_dict = {**model_dict, **hyps}
    with open(file_name,'wb') as f:
        pickle.dump(model_dict,f)

def save_ln_group(models, file_name):
    """
    Saves group of LNModels to file. Often models are trained using
    crossvalidation, which implicitely ties the models together as an
    individual unit.

    model: list of LNModel objects (see models.py)
    file_name: str
        path to save models to
    """
    model_dict = {
        "filts":[model.filt for model in models],
        "fits":[model.poly.coefficients for model in models],
        "spans":[model.span for model in models],
        "centers":[model.center for model in models],
        "norm_stats":[model.norm_stats for model in models],
        "cell_file":[model.cell_file for model in models],
        "cell_idx":[model.cell_idx for model in models],
    }
    with open(file_name,'wb') as f:
        pickle.dump(model_dict,f)
    

def save_checkpoint(save_dict, folder, exp_id, del_prev=False):
    """
    Saves the argued save_dict as a torch checkpoint.

    save_dict: dict
        all things to save to file
    folder: str
        path of folder to be saved to
    exp_id: str
        additional name to be prepended to path file string
    del_prev: bool
        if true, deletes the model_state_dict and optim_state_dict of
        the save of the previous file (used to save storage space)
    """
    if del_prev:
        temp = exp_id + "_epoch_" + str(save_dict['epoch']-1) + '.pt'
        prev_path = os.path.join(folder, temp)
        if os.path.exists(prev_path):
            device = torch.device("cpu")
            data = torch.load(prev_path, map_location=device)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key:
                    del data[key]
            torch.save(data, prev_path)
        elif save_dict['epoch'] != 0:
            print("Failed to find previous checkpoint", prev_path)
    temp = exp_id + '_epoch_' + str(save_dict['epoch'])
    path = os.path.join(folder, temp) + '.pt'
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder.

    folder: str
        path to the folder of interest
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    sort_key = lambda x: int(x.split(".")[-2].split("_")[-1])
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def get_model_folders(main_folder):
    """
    Returns a list of paths to the model folders contained within the
    argued main_folder

    main_folder - str
        path to main folder
    """
    folders = []
    main_folder = os.path.expanduser(main_folder)
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content:
                    folders.append(sub_d)
                    break
    sort_key = lambda x: int(x.split("/")[-1].split("_")[1])
    return sorted(folders, key=sort_key)

def load_checkpoint(checkpt_path):
    """
    Can load a specific model file both architecture and state_dict
    if the file contains a model_state_dict key, or can just load the
    architecture.

    checkpt_path: str
        path to checkpoint file
    """
    data = torch.load(checkpt_path, map_location=torch.device("cpu"))
    return data

def load_model(path):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        hyps = get_hyps(path)
        path = checkpts[-1]
    data = load_checkpoint(path)
    try:
        if 'hyps' in data:
            kwargs = data['hyps']
        elif 'model_hyps' in data:
            kwargs = data['model_hyps']
        elif hyps is not None:
            kwargs = hyps
        else:
            assert False, "Cannot find architecture arguments"
        model = globals()[data['model_type']](**kwargs)
    except Exception as e:
        print(e)
        print("Likely the checkpoint you are using is deprecated.")
    try:
        try:
            model.load_state_dict(data['model_state_dict'])
        except:
            print("Error loading state_dict, attempting fix..")
            sd = data['model_state_dict']
            sd_keys = list(sd.keys())
            m_keys = [name for name,_ in model.named_parameters()]
            for sk,mk in zip(sd_keys,m_keys):
                if sk != mk:
                    print("renaming {} to {}".format(sk,mk))
                    sd[mk] = sd[sk]
                    del sd[sk]
            model.load_state_dict(sd)
            print("Fix successful!")
    except KeyError as e:
        print("Failed to load state_dict. Key pairings:")
        for i,(sk,mk) in enumerate(zip(sd_keys,m_keys)):
            print(i,"State Dict:", sk)
            print(i,"Model:", mk)
    model.norm_stats = data['norm_stats']
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    hyps_json = os.path.join(folder, "hyperparams.json")
    hyps = tdrutils.load_json(hyps_json)
    return hyps
