import json
from typing import Any, Dict, Optional


def default_config() -> Dict[str, Any]:
	return {
		"train_gan": {
			"ModelId": "LB-EGAN_<Constructed Model Class ID>", #This should be specific to the constructed model and be denoted by the class ID.
			"data_path": "src/lb_egan/data/<Constructed Model Class ID>", #Should be the path of the original dataset.
			"n_epochs": 20000, #How many epochs to train.   
			"batch_size": 256, #Batch size for the training.
			"lr": 0.0002, #Learning rate for the training.
			"b1": 0.5, #Beta 1.
			"b2": 0.999, #Beta 2.
			"n_cpu": 8, #Number of CPUs to use for training.
			"latent_dim": 100, #Latent size for the generator.  
			"img_size": 128, #Image size for the generator.
			"channels": 3, #Channels for the generator.
			"sample_interval": 10 #Sample interval for the generator.
		},
		"generate_images": {
			"model_path": "src/lb_egan/models_zoo/generator_model.pth", #Should be the path of the trained generator model.
			"output_root": "src/lb_egan/outputs/generate_images", #Should be the path of the generated images.
			"class_name": "<Constructed Model Class ID>", #Constructed class ID, such as 02_Tapered
			"num_folds": 5, #How many folds to generate.
			"num_images": 100, #How many images to generate for each fold.
			"latent_size": 100, #Latent size for the generator.
			"start_sf": 0 #Starting index for the fold number.
		},
		"train_downstream": {
			"dataset_path": "src/lb_egan/outputs/generate_images", #Should be the path of the generated images.
			"output_path": "src/lb_egan/outputs/results", #Should be the path of the results.   
			"num_epochs": 20, #How many epochs to train.
			"optimizers": ["Adamax", "RMSprop", "SGD"], #Optimizers to use.
			"learning_rates": {
				"Adamax": [0.001], #The best learning rate is chosen by the notebook.
				"RMSprop": [0.0001], #The best learning rate is chosen by the notebook.
				"SGD": [0.001] #The best learning rate is chosen by the notebook.
			},
			"break_on": ["RMSprop", "SGD"] #The best optimizer is chosen by the notebook.
		}
	}


def load_config(path: Optional[str]) -> Dict[str, Any]:
	cfg = default_config()
	if not path:
		return cfg
	with open(path, 'r') as f:
		user_cfg = json.load(f)
	# shallow merge per section
	for section, values in user_cfg.items():
		if section in cfg and isinstance(values, dict):
			cfg[section].update(values)
		else:
			cfg[section] = values
	return cfg
