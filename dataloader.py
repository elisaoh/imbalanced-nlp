import warnings
warnings.filterwarnings('ignore')

import random
import time
import multiprocessing as mp
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd

import gluonnlp as nlp
import config as cfg




# The tokenizer takes as input a string and outputs a list of tokens.
tokenizer = nlp.data.SpacyTokenizer('en')

# `length_clip` takes as input a list and outputs a list with maximum length 500.
length_clip = nlp.data.ClipSequence(500)

# Helper function to preprocess a single data point
def preprocess(x, vocab):
	data, label = x
	label = int(label > 5)
	# A token index or a list of token indices is
	# returned according to the vocabulary.
	data = vocab[length_clip(tokenizer(data))]
	return data, label


# Helper function for getting the length 
def get_length(x):
    return float(len(x[0]))


def preprocess_dataset(dataset,vocab):

	start = time.time()
	# pool = mp.Pool(processes=2)
	# Each sample is processed in an asynchronous manner.
	dataset = gluon.data.SimpleDataset([preprocess(data,vocab) for data in dataset])
	lengths = gluon.data.SimpleDataset(map(get_length, dataset))
	# dataset = gluon.data.SimpleDataset(preprocess(dataset,vocab))
	# lengths = gluon.data.SimpleDataset(get_length(dataset))
	end = time.time()
	print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
	return dataset, lengths


def downsample_data(vocab, num_pos = 12500, num_neg = 12500, oversample = False):
	# Loading the dataset
	train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
	                               for segment in ('train', 'test')]
	# print('Tokenize using spaCy...')

	train_pos = random.sample(train_dataset[0:12500],num_pos)
	train_neg = random.sample(train_dataset[12500:],num_neg)

	if oversample:
		if num_pos > num_neg:
			train_neg = [random.choice(train_neg) for _ in range(num_pos)]
			# train_neg = random.choices(train_neg,k=num_pos)
		else:
			train_pos = [random.choice(train_pos) for _ in range(num_neg)]
			# train_pos = random.choices(train_pos,k=num_neg)

	train_dataset = train_pos + train_neg

		# Construct the DataLoader

	def get_dataloader():

	    # Pad data, stack label and lengths
	    batchify_fn = nlp.data.batchify.Tuple(
	        nlp.data.batchify.Pad(axis=0, ret_length=True),
	        nlp.data.batchify.Stack(dtype='float32'))
	    batch_sampler = nlp.data.sampler.FixedBucketSampler(
	        train_data_lengths,
	        batch_size=cfg.batch_size,
	        num_buckets=cfg.bucket_num,
	        ratio=cfg.bucket_ratio,
	        shuffle=True)
	    print(batch_sampler.stats())

	    # Construct a DataLoader object for both the training and test data
	    train_dataloader = gluon.data.DataLoader(
	        dataset=train_dataset,
	        batch_sampler=batch_sampler,
	        batchify_fn=batchify_fn)
	    test_dataloader = gluon.data.DataLoader(
	        dataset=test_dataset,
	        batch_size=cfg.batch_size,
	        shuffle=False,
	        batchify_fn=batchify_fn)
	    return train_dataloader, test_dataloader

	# Doing the actual pre-processing of the dataset
	train_dataset, train_data_lengths = preprocess_dataset(train_dataset,vocab)
	test_dataset, test_data_lengths = preprocess_dataset(test_dataset,vocab)

	# Use the pre-defined function to make the retrieval of the DataLoader objects simple
	train_dataloader, test_dataloader = get_dataloader()
	return train_dataloader, test_dataloader






