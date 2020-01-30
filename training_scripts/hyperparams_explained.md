This is a list of descriptions for each of the possible hyperparameters.

* `exp_name`: str
    * The name of the main experiment. Model folders will be saved within a folder of this name.
* `n_repeats`: int
    * The number of times to repeat any given hyperparameter set
* `save_every_epoch`: bool
    * A boolean determining if the model `state_dict` should be saved for every epoch, or only the most recent epoch.

* `model_type`: str
    * The string name of the main model class to be used for training. Options are each of the classes defined in `models.py`
* `n_layers`: int
    * the number of layers to be used. Only applies if using "VaryModel" model type. Must be greater than or equal to one.
* `bnorm`: bool
    * if true, model uses batch normalization where possible
* `bnorm_d`: int
    * determines if model uses batchnorm 2d or 1d. Only options are 1 and 2
* `bias`: bool
    * if true, model uses trainable bias parameters where possible
* `softplus`: bool
    * if true, a softplus activation function is used on the outputs of the model
* `chans`: list of ints
    * the number of channels to be used in the intermediary layers
* `ksizes`: list of ints
    * the kernel sizes of the convolutions corresponding to each layer
* `img_shape`: list of ints
    * the shape of the incoming stimulus to the model (do not include batchsize but do include depth of images)
* `stackconvs`: bool
    * if true, convolutions are trained using linear convolution stacking
* `convgc`: bool
    * if true, ganglion cell layer is convolutional

* `dataset`: str
    * the name of the dataset to be used for training. code assumes the datasets are located in `~/experiments/data/`. The dataset should be a folder that contains h5 files.
* `cells`: str or list of int
    * if string, "all" is only option which collects all cell recordings from the dataset. Otherwise, you can argue only the specific cells you would like to train with. See `datas.py` for more details.
* `stim_type`: str
    * the name of the h5 file (without the `.h5` extension) contained within the dataset folder
* `lossfxn`: str
    * The name of the loss function that should be used for training the model. Currently options are "PoissonNLLLoss" and "MSELoss"
* `log_poisson`: bool
    * only relevant if using "PoissonNLLLoss" function. If true, inputs are exponentiated before poisson loss is calculated.
* `shuffle`: bool
    * boolean determining if the order of samples with in a batch should be shuffled. This does not shuffle the sequence itself.

* `batch_size`: int
    * the number of samples to used in a single step of SGD
* `n_epochs`: int
    * the number of complete training loops through the data
* `lr`: float
    * the learning rate
* `l2`: float
    * the l2 weight penalty
* `l1`: float
    * the l1 activation penalty applied on the final outputs of the model.
* `noise`: float
    * the standard deviation of the gaussian noise layers
* `drop_p`: float
    * the dropout probability used in between linearly stacked convolutions. Only applies if `stackconvs` is set to true.
* `gc_bias`: bool
    * if true, final layer has a bias parameter
* `bn_moment`: float
    * the momentum of the batchnorm layers

* `prune`: bool
    * if true, layers are pruned
* `prune_layers`: list of str
    * enumerates the layers that should be pruned. If empty list, all intermediary convolutional layers are pruned.

