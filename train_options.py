import argparse

parser = argparse.ArgumentParser(description = 'Training')

# data parameters
parser.add_argument('data', metavar = 'DIR', help = 'path to dataset')
parser.add_argument('--model', default='', type=str, metavar = 'DIR', help = 'path to model')

# model parameters
parser.add_argument('--arch', metavar = 'ARCH', default = 'alexnet', 
					help = 'model architecture' + ' (default: alexnet)')
parser.add_argument('--lstm-layers', default=1, type=int, metavar='LSTM',
					help='number of lstm layers' + ' (default: 1)')
parser.add_argument('--hidden-size', default=512, type=int, metavar='HIDDEN',
					help='output size of LSTM hidden layers' + ' (default: 512)')
parser.add_argument('--fc-size', default=1024, type=int,
					help='size of fully connected layer before LSTM' + ' (default: 1024)')					

# train parameters
parser.add_argument('--epochs', default=100, type=int, metavar='N', 
					help='manual epoch number' + ' (default: 100)')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate' + ' (default: 0.001)')
parser.add_argument('--optim', default='sgd',type=str,
					help='optimizer' + ' (default: sgd)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum' + ' (default: 0.9)')
parser.add_argument('--lr-step', default=30, type=float,
					help='learning rate decay frequency' + ' (default: 30)')
parser.add_argument('--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size' + ' (default: 1)')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')					

# other parameters			
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')


