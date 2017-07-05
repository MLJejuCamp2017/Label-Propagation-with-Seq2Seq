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
batch_size = 32  # batch size
latent_dim = 256  # hidden layer dimension
num_block_sets = 6     # dilated blocks
filter_size = 3
learning_rate = 0.001

data = KOEN.merged_corpus(batch_size=batch_size, mode='test') # load parallel corpus
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
    emb_x = tf.Variable(tf.random_uniform([input_dim, latent_dim], -1.0, 1.0))
    #emb_y = tf.Variable(tf.random_uniform([input_dim, latent_dim], -1.0, 1.0))
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
#loss = tf.reduce_mean(loss)

# greedy search policy
label = tf.argmax(logits, 2)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# run graph for translating
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

t_avg = 1.
num_iter = 6
for i in range(num_iter):
    # generate output sequence
    num_train = batch_size
    for j in range(num_train):
        t_str = time.time()
        # predict character
        out1, out2 = sess.run([train, loss], 
                              {x: data.source[:batch_size], 
                               y: data.target[:batch_size], 
                               y0: np.zeros((batch_size,1)),
                               p_keep_conv: 0.9})
        t_elp = time.time() - t_str
        t_avg = 0.9*t_avg + 0.1*t_elp
        t_rem = t_avg * ((num_iter-i-1)*(num_train) + (num_train-j-1))
        print('[%d, %d: %.2fs, %dm]' % (i, j, t_elp, t_rem//60), np.mean(out2), end='\r')

    print()

    
    test_size = batch_size
    # initialize character sequence
    pred_prev = np.zeros((test_size, data.max_len)).astype(np.int32)
    pred = np.zeros((test_size, data.max_len)).astype(np.int32)

    # predict character
    x_enc = sess.run(enc, {x: data.source[:test_size]})

    for i in range(data.max_len):
        out = sess.run(label, {enc: x_enc, y_src: pred_prev,
                              p_keep_conv: 1.0})
        if i < data.max_len - 1:
            pred_prev[:, i+1] = out[:, i]
        pred[:, i] = out[:, i]

    # print result
    print('\nsources : --------------')
    data.print_index(data.source[:test_size], has_token=True)
    print('\ntargets : --------------')
    data.print_index(pred[:,:])
print()
