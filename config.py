import mxnet as mx


dropout = 0
language_model_name = 'standard_lstm_lm_200'
pretrained = True
learning_rate, batch_size = 0.005, 32
bucket_num, bucket_ratio = 10, 0.2
epochs = 5
grad_clip = None
log_interval = 100
context = mx.cpu()

num_pos = 1250
num_neg = 12500 

# random oversampling or not
oversample = True
