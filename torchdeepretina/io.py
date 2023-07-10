import torch
import pickle
from torchdeepretina.models import *
from torchdeepretina.custom_modules import *
import torchdeepretina.utils as utils
import torchdeepretina.pruning as tdrprune
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

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    x: str
    """
    splt = ("./"+x).split("/")[-1].split("_")
    for s in reversed(splt[1:]):
        try:
            return int(s)
        except:
            pass
    assert False

def get_model_folders(main_folder, incl_ext=False):
    """
    Returns a list of paths to the model folders contained within the
    argued main_folder

    main_folder - str
        path to main folder
    incl_ext: bool
        if true, includes full extension

    Returns:
        list of folders without full extension
    """
    folders = []
    main_folder = os.path.expanduser(main_folder)
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content or "hyperparams.txt" == content:
                    folders.append(sub_d.strip())
                    if incl_ext: 
                        folders[-1] = os.path.join(d, sub_d.strip())
                    break
    return sorted(folders, key=foldersort)

def load_checkpoint(path):
    """
    Can load a specific model file both architecture and state_dict
    if the file contains a model_state_dict key, or can just load the
    architecture.

    path: str
        path to checkpoint file
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        path = checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    return data

def load_model(path,verbose=True, ret_hyps=False):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints
    ret_hyps: bool
        if true, also returns hyperparameters dict
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        hyps = get_hyps(path)
        path = checkpts[-1]

    data = load_checkpoint(path)
    if 'hyps' in data:
        hyps = data['hyps']
        kwargs = hyps
    elif 'model_hyps' in data:
        hyps = data['model_hyps']
        kwargs = hyps
    elif hyps is not None:
        kwargs = hyps
    else:
        assert False, "Cannot find architecture arguments"
    try:
        hyps["dataset"] =   data['dataset']
        hyps["stim_type"] = data["stim_type"]
        hyps["norm_stats"] = data["norm_stats"]
        hyps["lossfxn"] = data["lossfxn"]
    except:
        for k in hyps.keys(): data[k] = hyps[k]
    if "img_shape" in data:
        hyps["img_shape"] = data["img_shape"]
    try:
        hyps["cells"] =     data['cells']
    except:
        pass

    try:
        model = globals()[data['model_type']](**kwargs)
    except Exception as e:
        print(e)
        print("Likely the checkpoint you are using is deprecated.")
    try:
        try:
            if "state_dict" in data:
                model.load_state_dict(data['state_dict'])
            else:
                model.load_state_dict(data['model_state_dict'])
        except:
            if verbose:
                print("Error loading state_dict, attempting fix..")
            sd = data['model_state_dict']
            sd_keys = list(sd.keys())
            m_keys = list(model.state_dict().keys())
            for sk,mk in zip(sd_keys,m_keys):
                if sk != mk:
                    if verbose:
                        print("renaming {} to {}".format(sk,mk))
                    sd[mk] = sd[sk]
                    del sd[sk]
            model.load_state_dict(sd)
            if verbose:
                print("Fix successful!")
    except KeyError as e:
        print("Failed to load state_dict. Key pairings:")
        for i,(sk,mk) in enumerate(zip(sd_keys,m_keys)):
            print(i,"State Dict:", sk)
            print(i,"Model:", mk)
    model.norm_stats = data.get('norm_stats', [0,1])
    model.zero_dict = utils.try_key(data,'zero_dict',dict())
    model.zero_bias = utils.try_key(hyps,'zero_bias',True)
    tdrprune.zero_chans(model, model.zero_dict, model.zero_bias)
    if ret_hyps:
        return model, hyps
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    try:
        hyps_json = os.path.join(folder, "hyperparams.json")
        hyps = utils.load_json(hyps_json)
    except:
        hyps_file = os.path.join(folder, "hyperparams.txt")
        hyps = dict()
        with open(hyps_file, "r") as f:
            for l in f.readlines():
                if l.strip() == l and ":" in l:
                    splt = l.split(": ")
                    hyps[splt[0]] = splt[1]
    return hyps

def get_next_exp_num(exp_path):
    """
    Finds the next open experiment id number.

    exp_path: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_path)
    exp_nums = set()
    for folder in folders:
        exp_num = foldersort(folder)
        exp_nums.add(exp_num)
    for i in range(len(exp_nums)):
        if i not in exp_nums and str(i) not in exp_nums:
            return i
    return len(exp_nums)

def exp_num_exists(exp_num, exp_name):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps):
    """
    Creates the save name for the model.

    hyps: dict
        keys:
            exp_name: str
            exp_num: int
            search_keys: str
    """
    save_folder = "{}/{}_{}".format(hyps['exp_name'],
                                    hyps['exp_name'],
                                    hyps['exp_num'])
    save_folder += hyps.get('search_keys', "")
    return os.path.abspath(
        os.path.join(hyps.get("save_root","./"), save_folder)
    )

