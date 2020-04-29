# CODE FOR MANTRA MODEL




## Dataset
We provide a dataloader for the KITTI dataset in *dataset.py*. This uses *kitti_dataset.json* for the trajectories and
the top view semantic maps which can be found in the *maps* folder.


## Installation
install required packages: 
```bash
pip install -r requirements.txt
```
---
## Trainings
To create a MANTRA model, 


Trainings can be monitored using tensorboard, logs are stored in the folder *runs/(runs-pretrain/runs-createMem/runs-IRM)*.
In pretrained_model folder, there are pretrained models of different step (autoencoder, writing controller, MANTRA model).

### Training encoder-decoder model (autoencoder)
```bash
train_ae.py
```
The autoencoder can be trained with the **train_ae.py** script. train_ae.py calls trainer_ae.py
The model will be saved into the folder *test/[current_date]*.
A pretrained model can be found in *pretrained_models/model_AE/*

### Training writing controller
```bash
train_controllerMem.py --model pretrained_autoencoder_model_path
```
The writing controller for the memory with autoencoder can be trained with **train_controllerMem.py**.
train_controllerMem.py calls trainer_controllerMem.py.
The path of a pretrained autoencoder model has to be passed to the script (it defaults to the pretrained model we provided).
A pretrained model (autoencoder + writing controller) can be found in *pretrained_models/model_controller/*

### Iterative Refinement Module (IRM) Training
```bash
train_IRM.py --model pretrained_autoencoder+controller_model_path
```

train_IRM.py calls trainer_IRM.py
The script trains the IRM module that generates the final prediction based on the decoded trajectory and the context map.
The paths of a pretrained autoencoder with writing controller model and populated memories have to be passed to the script (it defaults to the
pretrained models we provided).
A pretrained MANTRA model can be found in *pretrained_models/model_complete/*

---
## Test
```bash
test.py --model pretrained_complete_model_path   --withMRI True/False --memory_saved True/False
--memory_saved: if yes, the memories (past and memory) in /pretrained_models/memory_saved/ are loaded 
```
test.py calls evaluate_MemNet.py
This script generates metrics on the KITTI dataset using a trained models. We compute Average Displacement Error (ADE) and Horizon Error (Error@K).


### Command line arguments
```
    --host                         IP of the host server (default: 127.0.0.1)
    --port, -p                     TCP port to listen to (default: 2000)
    --mode                         Simulation mode. The user can choose: 'single_agent', 'multi_agent' or 'online'.
                                   In 'single_agent'/'multi_agent' mode, the system will generate predictions of EGO-VEHICLE/AGENTS NEAR EGO-VEHICLE 
                                   using the fixed memory filled with Carla Precog dataset.
                                   In 'online mode', the simulation starts with empty memory and while ego-vehicle is driving, 
                                   the system writes in memory the information to decrease prediction error. The error will be calculated 
                                   between the prediction that the system made at time-step T seconds with the real trajectory traveled by 
                                   ego-vehicle calculated at instant T+3 seconds.
    --res_width                    window resolution (width) (default: 800)
	--res_height				   window resolution (height) (default: 600)
    --number_agents                number of agents in the simulation. If number_agents=1, it will be only ego-centric vehicle (default: 10)
    --number_predictions           MANTRA model generates "number_predictions" predictions for each agent in the simulation. 
                                   If number_predictions=0, the simulation does not generate any prediction. (default: 5)

```

