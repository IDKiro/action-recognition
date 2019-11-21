import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset import loadedDataset
from model import LSTMModel


parser = argparse.ArgumentParser(description = 'Predicting')
parser.add_argument('--model', default='./save_model/model_best.pth.tar', help = 'path to model')
parser.add_argument('--data', default='./data/valid/', help = 'path to dataset')
args = parser.parse_args()

def predict(predict_loader, model):
	# switch to evaluate mode
	model.eval()
	
	result = [''] * 1500

	classes = []
	with open('./data/classes.txt', 'r') as cfile:
		classes = cfile.readlines()

	for i, (inputs, _, name) in enumerate(predict_loader):
		input_var = [input.cuda() for input in inputs]

		# compute output
		output = model(input_var)
		output = output[:, -1, :]
		_, pred = output.topk(1, 1, True, True)
		pred = pred.t()
		print('File Name: ' + name[0])
		print('Predict: ' + classes[pred[0][0].numpy()])
		
	# 	result[int(name[0])] = classes[pred[0][0].numpy()]

	# file= open('data/predict.txt', 'w')  
	# for fp in result:
	# 	file.write(str(fp))
	# file.close()


if __name__ == '__main__':

	predictdir = args.data

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.339, 0.224, 0.225])

	transform = (transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize]
									),
				transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor()]
									)
				)

	predict_loader = torch.utils.data.DataLoader(
		loadedDataset(predictdir, transform),
		batch_size=1, shuffle=False,
		num_workers=0, pin_memory=True)

	if os.path.exists(args.model):
		# load existing model
		model_info = torch.load(args.model)
		print("==> loading existing model '{}' ".format(model_info['arch']))
		original_model = models.__dict__[model_info['arch']](pretrained=False)
		model = LSTMModel(original_model, model_info['arch'],
			model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
		print(model)
		model.cuda()
		model.load_state_dict(model_info['state_dict'])
	else:
		print("Error: load model failed!")
		exit(1)

	predict(predict_loader, model)
