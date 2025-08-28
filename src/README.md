# LB-EGAN

This repository provides refactored Python modules derived from three original Jupyter notebooks:

- `LB-EGAN.ipynb` → `src/lb_egan/train_gan.py` (GAN training)
- `Image_Generator.ipynb` → `src/lb_egan/generate_images.py` (image generation with the trained GAN)
- `Classification.ipynb` → `src/lb_egan/train_downstream.py` (train a downstream model using GAN images)

Model architectures are copied verbatim and centralized in `src/lb_egan/models/architectures.py` to preserve behavior (layers, hyperparameters, and internals are unchanged).

## Installation

- Python 3.10+
- Install dependencies according to your environment (PyTorch, TorchVision, TorchMetrics, scikit-learn, seaborn, pandas, matplotlib).

## Package layout

- `src/lb_egan/models/architectures.py`: Verbatim model classes used by the notebooks.
- `src/lb_egan/train_gan.py`: Training script that reproduces the notebook flow one-to-one.
- `src/lb_egan/generate_images.py`: Image generation script for creating fold-structured datasets.
- `src/lb_egan/train_downstream.py`: Downstream classifier training with fold evaluation and confusion matrices.
- `src/lb_egan/config.py`: Centralized configuration with sane defaults and JSON override support.

## Default paths (inside src)

By default, all paths are set under the `src` tree to make the project self-contained:

- Training data for GAN: `src/lb_egan/data/augmented_data_all/02_Tapered`
- Pretrained generator weights (for generation): `src/lb_egan/models_zoo/generator_model.pth`
- Generated images root: `src/lb_egan/outputs/generate_images`
- Downstream results root: `src/lb_egan/outputs/results`

You can change any of these using a JSON config or CLI arguments (CLI always overrides config).

## Configuration

`src/lb_egan/config.py` contains default settings. You can create a JSON file to override any subset:

```json
{
  "train_gan": {
    "ModelId": "LB-EGAN_Tapered",
    "data_path": "src/lb_egan/data/augmented_data/02_Tapered",
    "n_epochs": 20000,
    "batch_size": 256,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 8,
    "latent_dim": 100,
    "img_size": 128,
    "channels": 3,
    "sample_interval": 10
  },
  "generate_images": {
    "model_path": "src/lb_egan/models_zoo/generator_model.pth",
    "output_root": "src/lb_egan/outputs/generate_images",
    "class_name": "Normal_Sperm",
    "num_folds": 5,
    "num_images": 100,
    "latent_size": 100,
    "start_sf": 0
  },
  "train_downstream": {
    "dataset_path": "src/lb_egan/outputs/generate_images",
    "output_path": "src/lb_egan/outputs/results",
    "num_epochs": 20,
    "optimizers": ["Adamax", "RMSprop", "SGD"],
    "learning_rates": {
      "Adamax": [0.001],
      "RMSprop": [0.0001],
      "SGD": [0.001]
    },
    "break_on": ["RMSprop", "SGD"]
  }
}
```

## Usage

- Train GAN (uses config defaults if not overridden):
```bash
python -m lb_egan.train_gan
```
Use custom config:
```bash
python -m lb_egan.train_gan --config /absolute/path/to/config.json
```

- Generate images with a pretrained generator:
```bash
python -m lb_egan.generate_images
```
Override via config or CLI:
```bash
python -m lb_egan.generate_images --config /absolute/path/to/config.json --output_root "src/lb_egan/outputs/generate_images"
```

- Train downstream classifier over generated folds:
```bash
python -m lb_egan.train_downstream
```
Or provide config/overrides:
```bash
python -m lb_egan.train_downstream --config /absolute/path/to/config.json --dataset_path "src/lb_egan/outputs/generate_images"
```

## Notes

- The GAN model classes reference `opt` internally as in the original notebook; the scripts assign `architectures.opt` at runtime before model construction to preserve behavior.
- Execution order and default hyperparameters match the notebooks. Any warnings or prints shown by the notebooks will appear similarly when running the scripts.

