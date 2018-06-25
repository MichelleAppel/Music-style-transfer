from torch.utils import data
import torch
import os
import math
import random
import numpy as np
from torchvision.transforms import Normalize


class DataSet(data.Dataset):
	"""Dataset class."""

	def __init__(self, data_dir, selected_attrs, mode, split):
		"""Initialize and preprocess the dataset."""
		self.data_dir = data_dir
		self.selected_attrs = selected_attrs or os.listdir(data_dir)
		self.mode = mode
		self.train_dataset = []
		self.test_dataset = []
		self.attr2idx = {}
		self.idx2attr = {}
		self.preprocess(split)

		if mode == 'train':
			self.length = len(self.train_dataset)
		else:
			self.length = len(self.test_dataset)

	def preprocess(self, split_percentage):
		"""Preprocess the attributes."""
		for i, attr_name in enumerate(self.selected_attrs):
			self.attr2idx[attr_name] = i
			self.idx2attr[i] = attr_name

			attr_dir = os.path.join(self.data_dir, attr_name)
			dataset = [(os.path.join(attr_dir, filename), attr_name) for filename in os.listdir(attr_dir)]

			random.seed(1234)
			random.shuffle(dataset)

			split = int(len(dataset) * split_percentage)
			self.train_dataset.extend(dataset[:split])
			self.test_dataset.extend(dataset[split:])

		print('Finished preprocessing the dataset...')


    def __getitem__(self, index):
        """Return one spectrogram and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filepath, label = dataset[index]
        # norm = Normalize(mean=[0.5], std=[0.5])
        # spectrogram = norm(torch.from_numpy(np.load(filepath)).unsqueeze(0))
        spectrogram = torch.from_numpy(np.load(filepath)).unsqueeze(0)
        return spectrogram, self.attr2idx[label], os.path.basename(filepath)

	def __len__(self):
		"""Return the number of files."""
		return self.length


def get_loader(data_dir, selected_attrs, split=0.8, batch_size=16, mode='train', num_workers=1):
	"""Build and return a data loader."""
	dataset = DataSet(data_dir, selected_attrs, mode, split)

	data_loader = data.DataLoader(
		dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
	return data_loader, dataset[0][0].shape[1:], len(dataset.selected_attrs)