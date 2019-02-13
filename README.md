# Torch Deep Retina

## Summary

### Current Models
- BatchNorm CNN (Lane and Niru)

### Current Training Scripts
- Salamander data

### Utils
- retio: saving and loading the model
- physiology: inspection, injection using hooks
- retinal phenomena/stimuli: code to generate stimulus and responses for retinal phenomena (lifted from drone)
- intracellular: methods to compare internal units with interneuron recordings
- batch compute: batch compute a model response (if the GPU runs out of memory)

## Setup
After cloning the repo and copying to your @deepretina.stanford.edu account. All requirements should be already installed globally in deepretina. If not, to install a package locally:
```sh
python3.6 -m pip install --user [package]
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






