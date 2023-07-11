# Torch Deep Retina
This repo contains the code for the paper [The dynamic neural code of the retina for natural scenes](https://www.biorxiv.org/content/10.1101/340943v5).

### Installation
This project is structured as a pip package. You will first need to install all the required dependencies.

If you are using `conda`, make sure you first install pip.

```
$ conda install pip
```

```
$ pip install -r requirements.txt
```

Then you will need to install this project.

```
$ pip install -e .
```

### Training
To train a model you will need to have data and a configuration json file, and optionally a config ranges json. [You can download the data here](https://doi.org/10.25740/rk663dm5577). The config json dictates the values of each of the training parameters that will be used for the training. See the [training_scripts readme](training_scripts/readme.md) for parameter details. The config-ranges json contains a subset of the config keys each indicating to a list of values that will be cycled through for training. Every combination of the config-ranges key value pairs will be scheduled for training. This allows for easy hyperparameter searches. For example, if `lr` is the only key in the config-ranges json, then trainings for each listed value of the learning rate will be queued and processed in order. If `lr` and `l2` each are in the config-ranges json, then every combination of the `lr` and `l2` values will be queued for training.

#### Data
You can find the data used in the paper at this link: [https://doi.org/10.25740/rk663dm5577](https://doi.org/10.25740/rk663dm5577)

If you are using Linux, the following command will probably work for you:

```
$ wget https://stacks.stanford.edu/file/druid:rk663dm5577/neural_code_data.zip
```

It will be easiest to then move the contents out of the folder named `ganglion_cell_data` into the top directory of the downloaded folder. We recommend that you move the ganglion cell data so that the interneuron data is located at the sam location as the ganglion cell data. To do this, do the following:

```
$ cd neural_code_data
$ mv ganglion_cell_data/* ./
```

This will allow you to argue the path to the downloaded directory as the entry for `datapath` in the configuration json:

```
{
    ...
    "datapath": "~/Downloads/neural_code_data/",
    ...
}
```

#### Running a Training
To run a training, navigate to the `training_scripts` folder:

```
$ cd training_scripts
```

Here you will want to create a new config.json file or use the existing. If you use the example config json you will likely run into issues with the dataset. Follow the section above to get data, and then be sure to update the config.json entry "datapath" with the path to the directory in which you downloaded and unzipped the data and moved the ganglion cell data to. So, for example, if you downloaded the data to your downloads folder `~/Downloads/`, unzipped it, and then moved the contents of `ganglion_cell_data` to the top downloaded directory, you would set `"datapath": "~/Downloads/neural_code_data/"`. If you did not move the ganglion cell data, you would instead do, `"datapath": "~/Downloads/neural_code_data/ganglion_cell_data/"`. We recommend that you move the ganglion cell data so that the interneuron data is located at the sam location as the ganglion cell data.

See the `readme.md` file in the `training_scripts` folder for details on all config options.

Then select the cuda device index you will want to use (in this case 0) and type the following command:

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_config.json
```
or
```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_config.json path_to_config-ranges.json
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

