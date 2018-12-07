import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

import dataset
from model import *


parser = argparse.ArgumentParser(description = 'Testing')


parser.add_argument('model', metavar = 'DIR', help = 'path to model')
parser.add_argument('data', metavar = 'DIR', help = 'path to dataset')


def validate(val_loader, model, criterion):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	for i, (input, target) in enumerate(val_loader):

		# target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)
		input_var, target_var = input_var.cuda(), target_var.cuda()

		# compute output
		output = model(input_var[0])
		weight = Variable(torch.Tensor(range(output.shape[0])) / sum(range(output.shape[0]))).cuda().view(-1,1).repeat(1, output.shape[1])
		output = torch.mul(output, weight)
		output = torch.mean(output, dim=0).unsqueeze(0)
		loss = criterion(output, target_var)
		losses.update(loss.item(), input.size(0))

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		print ('Test: [{0}/{1}]\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
			i, len(val_loader), loss=losses
			))

	return (top1.avg, top5.avg)

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def main():
	args = parser.parse_args()

	testdir = args.data

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

	test_loader = torch.utils.data.DataLoader(
		dataset.loadedDataset(testdir, transform),
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

	# loss criterion and optimizer
	criterion = nn.CrossEntropyLoss(reduction='none')
	criterion = criterion.cuda()

	prec1, prec5 = validate(test_loader, model, criterion)

	print("---------Test Result---------")
	print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1.item()))
	print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5.item()))
	print("-----------------------------")
		
if __name__ == '__main__':
	main()
