import warnings
warnings.filterwarnings('ignore')

import random
import time
import multiprocessing as mp
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd

from sklearn.metrics import classification_report as classification_report

import gluonnlp as nlp

from models import SentimentNet
from dataloader import *
from train import train
import config as cfg


random.seed(123)
np.random.seed(123)


lm_model, vocab = nlp.model.get_model(name=cfg.language_model_name,
                                      dataset_name='wikitext-2',
                                      pretrained=cfg.pretrained,
                                      ctx=cfg.context,
                                      dropout=cfg.dropout)

net = SentimentNet(dropout=cfg.dropout)
net.embedding = lm_model.embedding
net.encoder = lm_model.encoder
net.hybridize()
net.output.initialize(mx.init.Xavier(), ctx=cfg.context)

train_dataloader, test_dataloader = downsample_data(vocab, num_pos = cfg.num_pos, num_neg = cfg.num_neg, oversample = cfg.oversample)

train(net, cfg.context, cfg.epochs, train_dataloader, test_dataloader)