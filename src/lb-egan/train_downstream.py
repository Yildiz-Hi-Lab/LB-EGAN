import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn import metrics
import torchvision
from torchvision import datasets, models, transforms
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import argparse
from .config import load_config

cudnn.benchmark = True
plt.ion()


def prepare_data(path):
	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize([224,224]),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize([224,224]),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
	image_datasets = {x: datasets.ImageFolder(os.path.join(path, x), data_transforms[x]) for x in ['train', 'test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'test']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
	return dataloaders, dataset_sizes


def train_model(model, criterion, optimizer, scheduler, num_epochs, fold_num, dataloaders, dataset_sizes):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	since = time.time()
	best_acc = 0.0
	best_state = None
	for epoch in range(num_epochs):
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('-' * 10)
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			if phase == 'train':
				scheduler.step()
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_state = model.state_dict()
	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')
	if best_state is not None:
		model.load_state_dict(best_state)
	return model


def conf_mat(model, dataloaders):
	class_names = ["Normal", "Tapered","Pyriform", "Amorphous"]
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	fold_total = 0
	fold_correct = 0
	fold_true = []
	fold_predicted = []
	model.eval()
	with torch.no_grad():
		for inputs, labels in dataloaders['test']:
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs, 1)
			fold_total += labels.size(0)
			fold_correct += (predicted == labels).sum().item()
			fold_true.extend(labels.cpu().numpy())
			fold_predicted.extend(predicted.cpu().numpy())
	return(fold_total,fold_correct,fold_true,fold_predicted)


def restart_model(opti, learning_rate_selection):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Optional alternatives from the original notebook (uncomment to use):
	# model_ft = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
	# model_ft.name = 'efficientnet_v2_m'
	# model_ft = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
	# model_ft.name = 'efficientnet_v2_s'
	model_ft = models.densenet201(weights='IMAGENET1K_V1')
	model_ft.name = 'Densenet201'
	# num_ftrs = model_ft.AuxLogits.fc.in_features  # inception
	# model_ft.aux_logits = False
	# model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 18)  # inception
	# num_ftrs = model_ft.fc.in_features  # ResNet
	num_ftrs = model_ft.classifier.in_features  # DenseNet
	# num_ftrs = model_ft.classifier[1].in_features
	# num_ftrs = model_ft.heads[0].in_features
	# num_ftrs = model_ft.head.in_features
	# model_ft.classifier[1] = nn.Linear(num_ftrs, 3)   # ConvNeXt (try)
	# model_ft.head = nn.Linear(num_ftrs, 18)
	model_ft.fc = nn.Linear(num_ftrs, 4)
	model_ft = model_ft.to(device)
	criterion = nn.CrossEntropyLoss()
	if opti== 'Adamax':
		optimizer_ft = optim.Adamax(model_ft.parameters(), lr=learning_rate_selection)
	elif opti == 'SGD':
		optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate_selection, momentum=0.9)
	elif opti == 'RMSprop':
		optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=learning_rate_selection, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
	return(model_ft,criterion,optimizer_ft,exp_lr_scheduler)


def genel_fold_basarisi(fold_total,fold_correct,fold_true,fold_predicted, fold_accuracies, fold_true_labels, fold_predicted_labels):
	fold_accuracy = (fold_correct / fold_total) * 100
	fold_accuracies.append(fold_accuracy)
	fold_true_labels.extend(fold_true)
	fold_predicted_labels.extend(fold_predicted)
	return(fold_accuracies,fold_true_labels,fold_predicted_labels)


def fold_conf_mat(fold_true_labels,fold_predicted_labels,opti,learning_rate_selection,imageName,txtName,model_ft):
	class_names = ["Normal", "Tapered","Pyriform", "Amorphous"] 
	confusion_mtx = metrics.confusion_matrix(fold_true_labels, fold_predicted_labels)
	class_to_label = {'Normal': 0, 'Tapered': 1,'Pyriform':2, 'Amorphous': 3}
	df_cm = pd.DataFrame(confusion_mtx, index = [i for i in class_to_label], columns = [i for i in class_to_label])
	plt.figure(figsize = (10,10))
	plt.title(f'General Confusion Matrix for {model_ft.name} - {opti} - {learning_rate_selection}')
	result_confmat= sn.heatmap(df_cm, annot=True,cmap="OrRd",fmt="d",annot_kws={"size": 11},cbar=False)
	plt.ylabel('True labels',fontsize=12)
	plt.xlabel('Predicted labels',fontsize=12)
	figure = result_confmat.get_figure()
	figure.savefig(imageName, dpi=400)
	g_fold_class_report = metrics.classification_report(fold_true_labels, fold_predicted_labels, target_names=class_names, digits=4)
	print(f"Classification Report for Dataset for {model_ft.name} - {opti} - {learning_rate_selection}:\n{g_fold_class_report}")
	f=open(txtName,"w")
	f.write(g_fold_class_report)
	f.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--dataset_path", type=str, default=None)
	parser.add_argument("--output_path", type=str, default=None)
	parser.add_argument("--num_epochs", type=int, default=None)
	args = parser.parse_args()
	cfg = load_config(args.config)
	dc = cfg["train_downstream"]
	dataset_path = args.dataset_path if args.dataset_path is not None else dc["dataset_path"]
	output_path = args.output_path if args.output_path is not None else dc["output_path"]
	num_epochs = args.num_epochs if args.num_epochs is not None else dc["num_epochs"]
	for opti in dc["optimizers"]:
		if opti in dc.get("break_on", []):
			break
		Dizi = dc["learning_rates"].get(opti, [])
		for learning_rate_selection in Dizi:
			fold_accuracies = []
			fold_true_labels = []
			fold_predicted_labels = []
			print(f'Optimizer: {opti}')
			print(f'Learning Rate: {learning_rate_selection}')
			for fold_idx in range(1, 6):
				dataset_dir = dataset_path + f'/fold_{fold_idx}'
				dataloaders, dataset_sizes = prepare_data(dataset_dir)
				model_ft,criterion,optimizer_ft,exp_lr_scheduler=restart_model(opti,learning_rate_selection)
				imageName = output_path+'/'+ Path(dataset_path).parts[-1] +'_'+ model_ft.name + '_' + optimizer_ft.__class__.__name__ + '_' + str(learning_rate_selection) + '_Epoch_' + str(num_epochs) + '.png'
				txtName = output_path+'/'+ Path(dataset_path).parts[-1] +'_'+ model_ft.name + '_' + optimizer_ft.__class__.__name__ + '_' + str(learning_rate_selection) + '_Epoch_' + str(num_epochs)+'_Metrics.txt'
				model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs,fold_num=fold_idx, dataloaders=dataloaders, dataset_sizes=dataset_sizes)
				fold_total,fold_correct,fold_true,fold_predicted = conf_mat(model_ft,dataloaders)
				fold_accuracies,fold_true_labels,fold_predicted_labels = genel_fold_basarisi(fold_total,fold_correct,fold_true,fold_predicted, fold_accuracies, fold_true_labels, fold_predicted_labels)
				print(f"fold {fold_idx} analysis is finished")
			fold_conf_mat(fold_true_labels,fold_predicted_labels,opti,learning_rate_selection,imageName,txtName,model_ft)


if __name__ == "__main__":
	main()
