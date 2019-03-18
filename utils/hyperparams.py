import sys
import numpy as np

class HyperParams:
    def __init__(self, arg_hyps=None):
        
        hyp_dict = dict()
        hyp_dict['string_hyps'] = {
                    "exp_name":"default",
                    "data_set":"mnist",
                    "optim_type":'adam', # Options: rmsprop, adam
                    "resume_net_file": "_", # Use full path
                    "resume_optim_file": "_", # Use full path
                    "model_folder":"saved_models",
                    "log_folder":"saved_logs",
                    "loss_type":"_", # Currently only support crossentropy loss (see Trainer.get_loss_fxn)
                    "lr_schedule": "cosine_anneal",
                    }
        hyp_dict['int_hyps'] = {
                    "n_epochs": 30, # PPO update epoch count
                    "batch_size": 256, # PPO update batch size
                    'h_size':288,
                    'save_period':10, # Number of epochs per model save
                    'out_dim':10,
                    'lr_period':20,
                    'momentum_period':20,
                    }
        hyp_dict['float_hyps'] = {
                    "lr_high":float(1e-3),
                    "lr_low": float(1e-4),
                    "momentum_high":float(.9),
                    "momentum_low": float(.9),
                    "max_norm":.5,
                    "gauss_std":.1,
                    }
        hyp_dict['bool_hyps'] = {
                    "resume":False,
                    "bnorm": True,
                    }
        self.hyps = self.read_command_line(hyp_dict)
        if arg_hyps is not None:
            for arg_key in arg_hyps.keys():
                self.hyps[arg_key] = arg_hyps[arg_key]

    def read_command_line(self, hyps_dict):
        """
        Reads arguments from the command line. If the parameter name is not declared in __init__
        then the command line argument is ignored.
    
        Pass command line arguments with the form parameter_name=parameter_value
    
        hyps_dict - dictionary of hyperparameter dictionaries with keys:
                    "bool_hyps" - dictionary with hyperparameters of boolean type
                    "int_hyps" - dictionary with hyperparameters of int type
                    "float_hyps" - dictionary with hyperparameters of float type
                    "string_hyps" - dictionary with hyperparameters of string type
        """
        
        bool_hyps = hyps_dict['bool_hyps']
        int_hyps = hyps_dict['int_hyps']
        float_hyps = hyps_dict['float_hyps']
        string_hyps = hyps_dict['string_hyps']
        
        if len(sys.argv) > 1:
            for arg in sys.argv:
                arg = str(arg)
                sub_args = arg.split("=")
                if sub_args[0] in bool_hyps:
                    bool_hyps[sub_args[0]] = sub_args[1] == "True"
                elif sub_args[0] in float_hyps:
                    float_hyps[sub_args[0]] = float(sub_args[1])
                elif sub_args[0] in string_hyps:
                    string_hyps[sub_args[0]] = sub_args[1]
                elif sub_args[0] in int_hyps:
                    int_hyps[sub_args[0]] = int(sub_args[1])
    
        return {**bool_hyps, **float_hyps, **int_hyps, **string_hyps}

    def print(self):
        print("HyperParameter Dict:")
        for k in sorted(self.hyps.keys()):
            print(k,"-", self.hyps[k])
