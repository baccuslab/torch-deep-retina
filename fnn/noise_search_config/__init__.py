import os
from fnn.noise_search_config.default import _C

dir_path = os.path.dirname(os.path.abspath(__file__))

custom_config_files = {}

for entry in os.scandir(os.path.join(dir_path)):
    if entry.name.endswith("yaml"):
        cfg_name = "".join(entry.name.split(".")[:-1])
        custom_config_files[cfg_name] = entry.path

def get_default_cfg():
    return _C.clone()

def get_custom_cfg(name):
    cfg = get_default_cfg()
    cfg.merge_from_file(custom_config_files[name])
    return cfg