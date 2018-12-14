import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

import dataset
from model import *


parser = argparse.ArgumentParser(description = 'Predicting')


parser.add_argument('model', metavar = 'DIR', help = 'path to model')
parser.add_argument('data', metavar = 'DIR', help = 'path to dataset')


def predict(predict_loader, model):
	# switch to evaluate mode
	model.eval()
	
	result = [''] * 1500

	classes = []
	with open('./data/classes.txt', 'r') as cfile:
		classes = cfile.readlines()

	for i, (input, _, name) in enumerate(predict_loader):
		input_var = torch.autograd.Variable(input).cuda()

		# compute output
		output = model(input_var[0])	
		weight = Variable(torch.Tensor(range(output.shape[0])) / sum(range(output.shape[0]))).cuda().view(-1,1).repeat(1, output.shape[1])
		output = torch.mul(output, weight)
		output = torch.mean(output, dim=0).unsqueeze(0).data.cpu()
		_, pred = output.topk(1, 1, True, True)
		pred = pred.t()
		print('File Name: ' + name[0])
		print('Predict: ' + classes[pred[0][0].numpy()])
		
	# 	result[int(name[0])] = classes[pred[0][0].numpy()]

	# file= open('data/predict.txt', 'w')  
	# for fp in result:
	# 	file.write(str(fp))
	# file.close()

def main():
	args = parser.parse_args()

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
		dataset.loadedDataset(predictdir, transform),
		batch_size=1, shuffle=False,
		num_workers=8, pin_memory=True)

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
		return

	predict(predict_loader, model)

if __name__ == '__main__':
	main()
