import os
import shutil
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
import torch.nn.functional as F

from dataset import loadedDataset
from model import LSTMModel
from utils import AverageMeter


parser = argparse.ArgumentParser(description = 'Training')
parser.add_argument('--model', default='./save_model/', type=str, help = 'path to model')
parser.add_argument('--arch', default = 'resnet50', help = 'model architecture')
parser.add_argument('--lstm-layers', default=2, type=int, help='number of lstm layers')
parser.add_argument('--hidden-size', default=512, type=int, help='output size of LSTM hidden layers')
parser.add_argument('--fc-size', default=1024, type=int, help='size of fully connected layer before LSTM')					
parser.add_argument('--epochs', default=200, type=int, help='manual epoch number')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-step', default=100, type=float, help='learning rate decay frequency')
parser.add_argument('--batch-size', default=8, type=int, help='mini-batch size')						
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
args = parser.parse_args()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join('./save_model/', filename))
	if is_best:
		shutil.copyfile(os.path.join('./save_model/', filename), './save_model/model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
	if not epoch % args.lr_step and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


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


def train(train_loader, model, criterion, optimizer, epoch):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.train()	# switch to train mode

	for i, (inputs, target, _) in enumerate(train_loader):
		input_var = [input.cuda() for input in inputs]
		target_var = target.cuda()

		# compute output
		output = model(input_var)
		output = output[:, -1, :]
		loss = criterion(output, target_var)
		losses.update(loss.item(), 1)

		# compute accuracy
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0].item(), 1)
		top5.update(prec5[0].item(), 1)

		# zero the parameter gradients
		optimizer.zero_grad()

		# compute gradient
		loss.backward()
		optimizer.step()

		print('Epoch: [{0}][{1}/{2}]\t'
			'lr {lr:.5f}\t'
			'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
			'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
			epoch, i, len(train_loader),
			lr=optimizer.param_groups[-1]['lr'],
			loss=losses,
			top1=top1,
			top5=top5))


def validate(val_loader, model, criterion):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	for i, (inputs, target, _) in enumerate(val_loader):
		input_var = [input.cuda() for input in inputs]
		target_var = target.cuda()

		# compute output
		with torch.no_grad():
			output = model(input_var)
			output = output[:, -1, :]
			loss = criterion(output, target_var)
			losses.update(loss.item(), 1)

		# compute accuracy
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0].item(), 1)
		top5.update(prec5[0].item(), 1)

		print ('Test: [{0}/{1}]\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				i, len(val_loader),
				loss=losses,
				top1=top1,
				top5=top5))

	return (top1.avg, top5.avg)


if __name__ == '__main__':
	# Data Transform and data loading
	traindir = './data/train/'
	valdir = './data/valid/'

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

	train_dataset = loadedDataset(traindir, transform)
	val_dataset = loadedDataset(valdir, transform)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	if os.path.exists(os.path.join(args.model, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(args.model, 'checkpoint.pth.tar'))
		print("==> loading existing model '{}' ".format(model_info['arch']))
		original_model = models.__dict__[model_info['arch']](pretrained=False)
		model = LSTMModel(original_model, model_info['arch'],
			model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
		# print(model)
		model.cuda()
		model.load_state_dict(model_info['state_dict'])
		best_prec = model_info['best_prec']
		cur_epoch = model_info['epoch']
	else:
		if not os.path.isdir(args.model):
			os.makedirs(args.model)
		# load and create model
		print("==> creating model '{}' ".format(args.arch))
		original_model = models.__dict__[args.arch](pretrained=True)
		model = LSTMModel(original_model, args.arch,
			len(train_dataset.classes), args.lstm_layers, args.hidden_size, args.fc_size)
		# print(model)
		model.cuda()
		cur_epoch = 0

	# loss criterion and optimizer
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	optimizer = torch.optim.Adam([{'params': model.fc_pre.parameters()},
								{'params': model.rnn.parameters()},
								{'params': model.fc.parameters()}],
								lr=args.lr)

	best_prec = 0
	
	# Training on epochs
	for epoch in range(cur_epoch, args.epochs):

		optimizer = adjust_learning_rate(optimizer, epoch)

		print("---------------------------------------------------Training---------------------------------------------------")

		# train on one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		print("--------------------------------------------------Validation--------------------------------------------------")

		# evaluate on validation set
		prec1, prec5 = validate(val_loader, model, criterion)

		print("------Validation Result------")
		print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1))
		print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
		print("-----------------------------")

		# remember best top1 accuracy and save checkpoint
		is_best = prec1 > best_prec
		best_prec = max(prec1, best_prec)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'num_classes': len(train_dataset.classes),
			'lstm_layers': args.lstm_layers,
			'hidden_size': args.hidden_size,
			'fc_size': args.fc_size,
			'state_dict': model.state_dict(),
			'best_prec': best_prec,
			'optimizer' : optimizer.state_dict(),}, is_best)
