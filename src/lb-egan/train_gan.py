import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F
import torch
import torch.autograd as autograd

from torchmetrics.image.fid import FrechetInceptionDistance

import torchvision.utils as vutils
from PIL import Image

from tqdm import tqdm
from torchvision.models.inception import inception_v3

from .models import architectures
from .config import load_config


ModelId = "AggreGAN_Tapered"


def save_class_0_images(generator, latent_dim, save_dir=f"{ModelId}/{ModelId}_out", n_images=500):
	os.makedirs(save_dir, exist_ok=True)
	generator.eval()
	batch_size = 16
	num_batches = (n_images + batch_size - 1) // batch_size
	images_saved = 0
	with torch.no_grad():
		for _ in range(num_batches):
			z = torch.randn(batch_size, latent_dim).to(torch.device('cuda'))
			gen_imgs = generator(z)
			for img_idx in range(min(batch_size, n_images - images_saved)):
				img_path = os.path.join(save_dir, f"image_{images_saved + img_idx + 1}.jpg")
				vutils.save_image(gen_imgs[img_idx], img_path, normalize=True)
			images_saved += batch_size
			if images_saved >= n_images:
				break
	print(f"Total {n_images} images saved in '{save_dir}' directory.")
	generator.train()


def load_images_in_batches(folder, transform, batch_size=64):
	images = []
	for filename in os.listdir(folder):
		if filename.endswith(".jpg") or filename.endswith(".BMP"):
			img_path = os.path.join(folder, filename)
			try:
				img = Image.open(img_path).convert('RGB')
				img = transform(img)
				images.append(img)
			except Exception as e:
				print(f"Error loading image: {filename}, Error: {e}")
			if len(images) == batch_size:
				yield torch.stack(images)
				images = []
	if len(images) > 0:
		yield torch.stack(images)


def calculate_inception_score(fake_data_loader, splits=10):
	model = inception_v3(pretrained=True, transform_input=False).eval().to(torch.device('cuda'))
	preds = []
	with torch.no_grad():
		for images in tqdm(fake_data_loader, desc="Calculating Inception Score"):
			images = images.to(torch.device('cuda'))
			pred = F.softmax(model(images), dim=1).cpu().numpy()
			preds.append(pred)
	preds = np.concatenate(preds, axis=0)
	scores = []
	for i in range(splits):
		part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
		kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
		kl_div = np.mean(np.sum(kl_div, axis=1))
		scores.append(np.exp(kl_div))
	return np.mean(scores), np.std(scores)


def create_fake_data_loader(generator, latent_dim, batch_size=32, num_samples=1000):
	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	fake_images = []
	with torch.no_grad():
		for _ in range(num_samples // batch_size):
			z = torch.randn(batch_size, latent_dim).to(torch.device('cuda'))
			generated_images = generator(z).to(torch.device('cuda'))
			fake_images.append(transform(generated_images))
	fake_images = torch.cat(fake_images)
	return torch.utils.data.DataLoader(fake_images, batch_size=batch_size, shuffle=False)


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, X):
	alpha = FloatTensor(np.random.random(size=X.shape)).to(X.device)
	interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size(), device=X.device)))
	interpolates = Variable(interpolates, requires_grad=True)
	d_interpolates = D(interpolates)
	fake = Variable(FloatTensor(X.shape[0], 1).fill_(1.0), requires_grad=False)
	gradients = autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty


def sample_image(n_row, batches_done):
	z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
	gen_imgs = Ensemble_Generator(z)
	save_image(gen_imgs.data, f"{ModelId}/{ModelId}/%d.png" % batches_done, nrow=n_row, normalize=True)


def FIDCalc():
	transform = transforms.Compose([
		transforms.Resize((299,299)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	# use config-provided paths via globals set in main()
	original_dataset_paths = _fid_original_paths
	synthetic_dataset_path = _fid_synth_path
	fid = FrechetInceptionDistance(normalize=True).to(torch.device('cuda'))
	for original_dataset_path in original_dataset_paths:
		for batch in load_images_in_batches(original_dataset_path, transform):
			batch.size()
			fid.update(batch.to(torch.device('cuda')), real=True)
	for batch in load_images_in_batches(synthetic_dataset_path, transform):
		fid.update(batch.to(torch.device('cuda')), real=False)
	fid_score = fid.compute()
	return fid_score


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--n_epochs", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=None)
	parser.add_argument("--lr", type=float, default=None)
	parser.add_argument("--b1", type=float, default=None)
	parser.add_argument("--b2", type=float, default=None)
	parser.add_argument("--n_cpu", type=int, default=None)
	parser.add_argument("--latent_dim", type=int, default=None)
	parser.add_argument("--img_size", type=int, default=None)
	parser.add_argument("--channels", type=int, default=None)
	parser.add_argument("--sample_interval", type=int, default=None)
	args, _ = parser.parse_known_args()
	cfg = load_config(args.config)
	gan_cfg = cfg["train_gan"]
	global ModelId
	ModelId = gan_cfg.get("ModelId", ModelId)
	os.makedirs(f"{ModelId}/{ModelId}", exist_ok=True)
	# Build opt using config with CLI overrides when provided
	class OptObj:
		pass
	opt_local = OptObj()
	opt_local.n_epochs = args.n_epochs if args.n_epochs is not None else gan_cfg["n_epochs"]
	opt_local.batch_size = args.batch_size if args.batch_size is not None else gan_cfg["batch_size"]
	opt_local.lr = args.lr if args.lr is not None else gan_cfg["lr"]
	opt_local.b1 = args.b1 if args.b1 is not None else gan_cfg["b1"]
	opt_local.b2 = args.b2 if args.b2 is not None else gan_cfg["b2"]
	opt_local.n_cpu = args.n_cpu if args.n_cpu is not None else gan_cfg["n_cpu"]
	opt_local.latent_dim = args.latent_dim if args.latent_dim is not None else gan_cfg["latent_dim"]
	opt_local.img_size = args.img_size if args.img_size is not None else gan_cfg["img_size"]
	opt_local.channels = args.channels if args.channels is not None else gan_cfg["channels"]
	opt_local.sample_interval = args.sample_interval if args.sample_interval is not None else gan_cfg["sample_interval"]
	print(opt_local.__dict__)

	# Set global FID paths from config defaults
	global _fid_original_paths, _fid_synth_path
	_fid_original_paths = [gan_cfg["data_path"]]
	_fid_synth_path = f"{ModelId}/{ModelId}_out"

	cuda = True if torch.cuda.is_available() else False

	global Ensemble_Generator
	global generatorDG, discriminatorDG
	global generatorDC, discriminatorDC
	global crit, lambda_gp
	global FloatTensor, LongTensor, opt

	opt = opt_local
	Ensemble_Generator = architectures.Ensemble_Generator(
		latent_dim=opt.latent_dim,
		img_size=opt.img_size,
		channels=opt.channels
	)
	if cuda:
		Ensemble_Generator.cuda()

	lambda_gp = 0.1

	generatorDG = architectures.GeneratorDG(
		latent_dim=opt.latent_dim,
		img_size=opt.img_size,
		channels=opt.channels
	)
	discriminatorDG = architectures.DiscriminatorDG(
		img_size=opt.img_size,
		channels=opt.channels
	)
	if cuda:
		generatorDG.cuda()
		discriminatorDG.cuda()
	crit = torch.nn.BCELoss()
	if cuda:
		crit.cuda()

	generatorDG.apply(weights_init_normal)
	discriminatorDG.apply(weights_init_normal)

	generatorDC = architectures.GeneratorDC(
		latent_dim=opt.latent_dim,
		img_size=opt.img_size,
		channels=opt.channels
	)
	discriminatorDC = architectures.DiscriminatorDC(
		img_size=opt.img_size,
		channels=opt.channels
	)
	if cuda:
		generatorDC.cuda()
		discriminatorDC.cuda()
	generatorDC.apply(weights_init_normal)
	discriminatorDC.apply(weights_init_normal)

	data_path = gan_cfg["data_path"]
	dataset = ImageFolder(
		root=data_path,
		transform=transforms.Compose([
			transforms.Resize((opt.img_size, opt.img_size)),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5], [0.5])
		])
	)
	dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

	optimizer_DG = torch.optim.Adam(generatorDG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optimizer_DD = torch.optim.Adam(discriminatorDG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

	optimizer_DC = torch.optim.Adam(generatorDC.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optimizer_DC = torch.optim.Adam(discriminatorDC.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

	FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

	fid_p = 9999

	for epoch in range(opt.n_epochs):
		for i, (imgs, labels) in enumerate(dataloader):
			batch_size = imgs.shape[0]
			valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
			fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
			real_imgs = Variable(imgs.type(FloatTensor))
			z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

			optimizer_DG.zero_grad()
			gen_imgsDG = generatorDG(z)
			validity = discriminatorDG(gen_imgsDG)
			dg_loss = crit(validity, valid)
			dg_loss.backward()
			optimizer_DG.step()

			optimizer_DD.zero_grad()
			validity_real = discriminatorDG(real_imgs)
			d_real_loss = crit(validity_real, valid)
			validity_fake = discriminatorDG(gen_imgsDG.detach())
			d_fake_loss = crit(validity_fake, fake)
			dd_loss = (d_real_loss + d_fake_loss) / 2
			gradient_penalty = compute_gradient_penalty(discriminatorDG, real_imgs.data)
			dd_loss = gradient_penalty + dd_loss
			dd_loss.backward()
			optimizer_DD.step()

			optimizer_DC.zero_grad()
			gen_imgsDC = generatorDC(z)
			DCg_loss = crit(discriminatorDC(gen_imgsDC), valid)
			DCg_loss.backward()
			optimizer_DC.step()

			optimizer_DC.zero_grad()
			DCreal_loss = crit(discriminatorDC(real_imgs), valid)
			DCfake_loss = crit(discriminatorDC(gen_imgsDC.detach()), fake)
			DCd_loss = (DCreal_loss + DCfake_loss) / 2
			DCd_loss.backward()
			optimizer_DC.step()

			w1 = 1 / DCd_loss
			w2 = 1 / dd_loss
			total_weight = w1 + w2
			w1 /= total_weight
			w2 /= total_weight

			weighted_state_dict = {}
			for key in Ensemble_Generator.state_dict().keys():
				weighted_state_dict[key] = w1 * generatorDC.state_dict()[key] + w2 * generatorDG.state_dict()[key]
			Ensemble_Generator.load_state_dict(weighted_state_dict)
			generatorDC.load_state_dict(Ensemble_Generator.state_dict())
			generatorDG.load_state_dict(Ensemble_Generator.state_dict())

			print(
				"[Epoch %d/%d] [Batch %d/%d] [DCD loss: %f] [DD loss: %f] [DCG loss: %f] [DG loss: %f]"
				% (epoch, opt.n_epochs, i, len(dataloader), DCd_loss.item(), dd_loss.item(), DCg_loss.item(), dg_loss.item())
			)

			batches_done = epoch * len(dataloader) + i
			epoch_done = epoch
			if epoch_done % opt.sample_interval == 0 and i == 2:
				sample_image(n_row=4, batches_done=epoch_done)
				save_class_0_images(generator=Ensemble_Generator, latent_dim=opt.latent_dim)
				fid_c = FIDCalc()
				f = open(f"{ModelId}/FID_Scores_{ModelId}.txt", "a")
				f.write("\n" + str(fid_c))
				if fid_p > fid_c:
					save_dir = f"{ModelId}/{ModelId}_Out"
					os.makedirs(save_dir, exist_ok=True)
					generator_path = os.path.join(save_dir, f'generator_model.pth')
					torch.save(Ensemble_Generator.state_dict(), generator_path)
					fid_p = fid_c
					print(fid_p)
				fake_loader = create_fake_data_loader(Ensemble_Generator, latent_dim=100)
				is_mean, is_std = calculate_inception_score(fake_loader)
				print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
				I = open(f"{ModelId}/IS_{ModelId}.txt", "a")
				I.write("\n" + str(is_mean))


if __name__ == "__main__":
	main()
