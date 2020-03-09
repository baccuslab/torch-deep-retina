# Torch Deep Retina
This repo contains the code used to produce the figures in [The dynamic neural code of the retina for natural scenes](https://www.biorxiv.org/content/10.1101/340943v5).

### Training
To train a model you will need to have a hyperparameters json and a hyperranges json. The hyperparameters json details the values of each of the training parameters that will be used for the training. See the [training_scripts readme](training_scripts/readme.md) for parameter details. The hyperranges json contains a subset of the hyperparameter keys each coupled to a list of values that will be cycled through for training. Every combination of the hyperranges key value pairs will be scheduled for training. This allows for easy hyperparameter searches. For example, if `lr` is the only key in the hyperranges json, then trainings for each listed value of the learning rate will be queued and processed in order. If `lr` and `l2` each are in the hyperranges json, then every combination of the `lr` and `l2` values will be queued for training.

To run a training session, navigate to the `training_scripts` folder:

```
$ cd training_scripts
```

And then select the cuda device index you will want to use (in this case 0) and type the following command:

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_hyperparameters.json path_to_hyperranges.json
```
### Analysis
##### Full Model Analyses
It is perhaps best for each user to do their own analyses. This will reduce the risk of misinterpretations of results. There is, however, an automated analysis pipeline that can be used if your model falls in the categories below. To use this automated analysis tool, simply argue the name of the main experiment folder to the `analysis_pipeline.py` script within the `training_scripts/` folder:

```
$ python3 analysis_pipeline.py path_to_experiment_folder
```

This will create figures in each of the model folders contained within the experiment folder. It will also create a csv in the experiment folder that details a number of high level results for each of the models that can be used with pandas.

##### Interneuron Analyses
For doing interneuron analyses, it is best to use the `get_intr_cors()` function in the `torchdeepretina/analysis.py` package. This will return a pandas dataframe with the correlation for each interneuron cell recording with each unit in the model. This is automatically generated if using `analysis_pipeline.py`.

### Current Models
- BNCNN: the model architecture used for the model in [Deep Learning Models of the Retinal Response to Natural Scenes](https://papers.nips.cc/paper/6388-deep-learning-models-of-the-retinal-response-to-natural-scenes).
- LinearStackedBNCNN: the training architecture used for the model in [The dynamic neural code of the retina for natural scenes](https://www.biorxiv.org/content/10.1101/340943v5)
- Vary Model: a class that can take on the BNCNN (slight difference in that the batchnorm layers are constrained to be positive) or the LinearStackedBNCNN architectures.
- RetinotopicModel: a class that is similar to the VaryModel, but uses a one-hot layer to allow for fully convolutional training

## Setup
After cloning the repo and copying to your @deepretina.stanford.edu account, install all necessary packages locally:
```sh
python3.6 -m pip install --user -r requirements.txt
```
Next you will to install torchdeepretina. Run the following:
```sh
python3.6 -m pip install --user -e .
```

Check GPU status:
```sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=[0 or 1]
```

Use tmux for training:
```sh
tmux new -t training
cd training_scripts
```
Train your model!
```sh
python3.6 15-10-07_salamander.py --save="~/yoursavedirectory/"
```
You can use flags --batch, --lr, --l1, --l2, --epochs, and --shuffle to play with hyperparameters.

Running jupyter notebook on deepretina
```sh
jupyter-notebook --no-browser --port=8800
```
In your local terminal:
```sh
ssh -fNL 8800:localhost:8800 username@deepretina.stanford.edu
```

Copy and paste the link into your browser. 
Check out the BNCNN_Analysis jupyter notebook for an example of how to load a model, plot metrics, and analyze with intracellular recordings.






