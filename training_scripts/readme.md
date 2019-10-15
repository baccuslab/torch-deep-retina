To train a model already defined in `torchdeepretina/models.py`, you can simply run the following line at the terminal from the `training_scripts` folder:

    $ python3 main.py hyperparams.json hyperranges.json

The hyperparams.json should have a list of all the desired user setting for the training. The hyperranges.json should have the values wished to be searched over for each desired user setting key.

### The hyper keys
exp\_name - string name of the folder that all searches will be saved to (can be new name or existing)
model\_type - string of the class of the desired model architecture defined in `models.py`
dataset - string name of the desired dataset (15-10-07, 15-11-21a, 15-10-21b)
cells - list of desired ganglion cell indices or string "all" if all are desired
stim\_type - string name of stimulus type (naturalscene or whitenoise)
shuffle - bool dictating whether data is shuffled during training
starting\_exp\_num - an ID number is given to each combination of hyperparameters. It is possible to duplicate this ID number. If this field is set to null, the next available ID number is used.
lossfxn - string name of pytorch nn lossfunction (PoissonNLLLoss or MSELoss)
log\_poisson - bool used in PoissonNLLLoss (see pytorch documentation)
softplus - bool if true a Softplus activation is used at the final layer of the model
batch\_size - int number of elements per training batch
n\_epochs - int number of training epochs (full training cycles)
lr - float learning rate
l1 - float l1 loss applied to final activation layer
l2 - float weight decay applied to all parameters in model
noise - float the standard deviation for the gaussian noise layers
bias - bool use a bias vector for the weight layers
bnorm - bool use batchnorm layers
drop\_p - float include dropout at inner layers of stacked convolution with probability drop\_p to be dropped
linear\_bias - bool use bias at final linear layer (overrides `bias`)
bnorm\_momentum - float momentum parameter for the normalization statistics of the batchnorm layers
chans - list of ints of the number of channels for each layer
ksizes - list of ints of the filter size for each layer
img\_shape - the desired inital image shape
stackconvs - bool for some models can easily swap conv2d layers for LinearStackedConv2d layers
convgc - bool if true, ganglion cell layer (final layer) is trained as a convolution layer
one2one - determines if LinearStackedConv2d layer has cross connections at intermediate layers
skip\_nums - list of ints of experiment numbers to skip. Allows you to skip certain hyperparameter combinations in a hypersearch


