import os
import torch
from PIL import Image

from .models.architectures import Generator
import argparse
from .config import load_config


def train_and_generate_images(model_path, output_root, class_i, num_folds=5, num_images=100, latent_size=100, start_sf=0):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Generator()
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()
	for fold in range(1, num_folds + 1):
		fold_dir = os.path.join(output_root, f"fold_{fold}", "train", f"{class_i}")
		os.makedirs(fold_dir, exist_ok=True)
		print(f"Fold {fold}: Images are saved to {fold_dir}...")
		with torch.no_grad():
			for i in range(num_images):
				test_input = torch.randn(1, latent_size, device=device)
				prediction = model(test_input).squeeze(0)
				image_array = (prediction.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
				image = Image.fromarray(image_array.astype('uint8'))
				image_path = os.path.join(fold_dir, f'generated_image_{start_sf + i + 1}.jpg')
				image.save(image_path)
		start_sf += num_images
		print(f"Fold {fold} is finished.")


def main():
	# Defaults mirror the notebook; can be overridden via CLI or config file
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--output_root", type=str, default=None)
	parser.add_argument("--class_name", type=str, default=None)
	parser.add_argument("--num_folds", type=int, default=None)
	parser.add_argument("--num_images", type=int, default=None)
	parser.add_argument("--latent_size", type=int, default=None)
	parser.add_argument("--start_sf", type=int, default=None)
	args = parser.parse_args()
	cfg = load_config(args.config)
	gc = cfg["generate_images"]
	model_path = args.model_path if args.model_path is not None else gc["model_path"]
	output_root = args.output_root if args.output_root is not None else gc["output_root"]
	class_name = args.class_name if args.class_name is not None else gc["class_name"]
	num_folds = args.num_folds if args.num_folds is not None else gc["num_folds"]
	num_images = args.num_images if args.num_images is not None else gc["num_images"]
	latent_size = args.latent_size if args.latent_size is not None else gc["latent_size"]
	start_sf = args.start_sf if args.start_sf is not None else gc["start_sf"]
	train_and_generate_images(
		model_path=model_path,
		output_root=output_root,
		class_i=f"{class_name}",
		num_folds=num_folds,
		num_images=num_images,
		latent_size=latent_size,
		start_sf=start_sf,
	)


if __name__ == "__main__":
	main()
