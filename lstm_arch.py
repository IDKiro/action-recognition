import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable


class FineTuneLstmModel(nn.Module):
	def __init__(self, original_model, arch, num_classes, lstm_layers, hidden_size, fc_size):
		super(FineTuneLstmModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.fc_size = fc_size

		# select a base model
		if arch.startswith('alexnet'):
			self.features = original_model.features
			self.fc_pre = nn.Sequential(nn.Linear(256 * 6 * 6, fc_size), nn.Dropout())

			self.rnn = nn.LSTM(input_size = fc_size,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'alexnet_lstm'

		else:
			raise Exception("This architecture has not been supported yet")

	def init_hidden(self, num_layers, batch_size):
		return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda(),
				Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda())

	def forward(self, inputs, hidden=None, steps=0):
		length = len(inputs)
		fs = Variable(torch.zeros(length, self.rnn.input_size)).cuda()
		for i in range(length):
			# print self.features(inputs[i].unsqueeze(0)).shape
			f = self.features(inputs[i].unsqueeze(0))
			f = f.view(f.size(0), -1)
			f = self.fc_pre(f)
			fs[i] = f
		fs = fs.unsqueeze(0)

		outputs, hidden = self.rnn(fs, hidden)
		outputs = self.fc(outputs[0])
		return outputs, hidden
