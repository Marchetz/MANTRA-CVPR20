# Installation

A conda environment is provided to run the code. It can be created by importing the configuration file store in
**environment_memnet.yml**

Create the environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate conda_env
```

# Training encoder-decoder model (autoencoder)
```bash
train_ae.py
```
The model can be trained with the **train_ae.py** script.
The model will be saved into the folder *test/[current_date]*.

Training can be monitored using tensorboard, logs are stored in the folder *runs-pretrain*.

A pretrained model can be found in *pretrained_models/model_AE2019-05-23 15:50:06*

# Training writing controller
```bash
train_controllerMem.py --model pretrained_autoencoder_model_path
```
The script trains the writing controller for the memory.
The path of a pretrained autoencoder model has to be passed to the script (it defaults to the pretrained model we provided).


# Iterative Refinement Module (IRM) Training
```bash
train_IRM.py --model pretrained_autoencoder_model_path
```
The script trains the IRM module that generates the final prediction based on the decoded trajectory and the context map.

The paths of a pretrained autoencoder model and populated memories have to be passed to the script (it defaults to the
pretrained models we provided).

Training can be monitored with tensorboard, logs are stored in the folder *runs-IRM*.

A pretrained model can be found in *pretrained_models/model_decoder_FT/model_FTdecoder2019-09-19 09:48:05*

file_model: models/model_memory_IRM.py

# Test
```bash
test.py --model pretrained_complete_model_path   --withMRI True/False --memory_saved True/False
--memory_saved: if yes, the memory in /pretrained_models/memory_saved/ are loaded 
```
This script generates metrics on the KITTI dataset using a trained models. We compute Average Displacement Error (ADE)
and Horizon Error (Error@K).

# Dataset
We provide a dataloader for the KITTI dataset in *dataset.py*. This uses top view semantic maps which can be found in
the *maps* folder.  
