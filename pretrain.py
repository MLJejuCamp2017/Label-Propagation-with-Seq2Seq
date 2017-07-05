import time
import numpy as np
import tensorflow as tf
from data_koen import KOEN
from bytenet_ops import *

# set log level to debug
tf.logging.set_verbosity(tf.logging.DEBUG)


#
# hyper parameters
#
batch_size = 32   # batch size
latent_dim = 256   # hidden layer dimension
num_block_sets = 3     # dilated blocks
filter_size = 3

learning_rate = 0.00005
#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.96, staircase=True)

import pickle
from glob import glob
if glob('logs/pretrain/data.pickle'):
    with open('logs/pretrain/data.pickle', 'rb') as f:
        data = pickle.load(f)
else:
    data = KOEN.merged_corpus(batch_size=batch_size, mode='train') # load parallel corpus
    # reduce data
    num_batch_pre = 1895
    data.source = data.source[:num_batch_pre*batch_size]
    data.target = data.target[:num_batch_pre*batch_size]
    data.num_batch = num_batch_pre
    data.num_data = num_batch_pre*batch_size
    with open('logs/pretrain/data.pickle', 'wb') as f:
        pickle.dump(data, f)

input_dim = data.voca_size
input_max_len = data.max_len

#
# inputs
#
with tf.name_scope('input'):
    # place holders
    x = tf.placeholder(dtype='int32', shape=[None, input_max_len])
    y = tf.placeholder(dtype='int32', shape=[None, input_max_len])

    # make embedding matrix for source and target
    emb_x = tf.Variable(tf.random_uniform([input_dim,latent_dim], -1.0, 1.0))
    emb_y = emb_x

    # shift target for training source
    y0 = tf.placeholder(dtype='int32', shape=[None, 1])
    y_src = tf.concat([y0, y[:, :-1]], 1)

#
# encode graph ( atrous convolution )
#

# embed table lookup
enc = tf.nn.embedding_lookup(emb_x, x)
enc = encoder(enc, filter_size=filter_size, num_block_sets=num_block_sets)

#
# decode graph ( causal convolution )
#

# loop dilated causal conv block
dec = tf.concat([enc, tf.nn.embedding_lookup(emb_y, y_src)], 2)
dec = decoder(dec, filter_size=filter_size, num_block_sets=num_block_sets)


with tf.name_scope('output'):
    # additional convolution and relu
    out = layer_norm(dec, causal=True)
    out = tf.nn.relu(out)
    out_dim = out.get_shape().as_list()[-1] # latent_dim * 2
    w_shape = [1, out_dim, out_dim]
    w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
    w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
    out = tf.nn.convolution(out, w, padding='SAME')

    # final fully convolution layer for softmax
    logits = layer_norm(out, causal=True)
    logits = tf.nn.relu(logits)
    
    p_keep_conv = tf.placeholder(tf.float32) # should be 0.9
    logits = tf.nn.dropout(logits, p_keep_conv)
    
    w_shape = [1, out_dim, input_dim]
    w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
    w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
    logits = tf.nn.convolution(logits, w, padding='SAME')

# cross entropy loss with logit and mask
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.identity(loss)
loss *= tf.cast(tf.not_equal(y, tf.zeros_like(y)), loss.dtype)
loss = tf.reduce_mean(loss)

# greedy search policy
label = tf.argmax(logits, 2)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

saver = tf.train.Saver(max_to_keep=20)
# run graph for translating
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# init session vars
checkpoint_dir = 'logs/pretrain/'
latest_model_path = tf.train.latest_checkpoint(checkpoint_dir)
cnt = 0

if latest_model_path:
    print('model exsists: ', latest_model_path)
    saver.restore(sess, latest_model_path)

    import re
    cnt = int(re.search('pretrain-(\w+).ckpt', latest_model_path).group(1)) + 1

t_avg = 1.
num_iter = 32
l_avg = 1.
for i in range(num_iter):
    # generate output sequence
    N = len(data.source)
    ids = np.arange(N)
    np.random.shuffle(ids)
    data.source = [data.source[id] for id in ids]
    data.target = [data.target[id] for id in ids]

    for j in range(data.num_batch):
        t_str = time.time()
        # predict character
        out1, out2 = sess.run([train, loss], 
                              {x: data.source[j*batch_size:(j+1)*batch_size], 
                               y: data.target[j*batch_size:(j+1)*batch_size], 
                               y0: np.zeros((batch_size,1)),
                               p_keep_conv: 0.9})
        t_elp = time.time() - t_str
        t_avg = 0.9*t_avg + 0.1*t_elp
        l_avg = 0.99*l_avg + 0.01*np.mean(out2)
        t_rem = t_avg * ((num_iter-i-1)*(data.num_batch) + (data.num_batch-j-1))
        print('[%d, %d: %.2fs, %dm]' % (i, j, t_elp, t_rem//60), l_avg, end='\r')

    save_path = saver.save(sess, (checkpoint_dir + "pretrain-%d.ckpt") % (i+cnt))
    print()

    
    test_size = 10 
    # initialize character sequence
    pred_prev = np.zeros((test_size, data.max_len)).astype(np.int32)
    pred = np.zeros((test_size, data.max_len)).astype(np.int32)

    # output of encoder
    x_enc = sess.run(enc, {x: data.source[:test_size]})

    # generate output sequence
    for i in range(data.max_len):
        # predict character
        out = sess.run(label, {enc: x_enc, y_src: pred_prev,
                              p_keep_conv: 1.0})
        # update character sequence
        if i < data.max_len - 1:
            pred_prev[:, i+1] = out[:, i]
        pred[:, i] = out[:, i]

    # print result
    print('\nsources : --------------')
    data.print_index(data.source[:test_size], has_token=True)
    print('\ntargets : --------------')
    data.print_index(pred[:,:])
print()
