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
latent_dim = 400   # hidden layer dimension
num_block_sets = 6     # dilated blocks
filter_size = 3
learning_rate = 0.0003

data = KOEN(batch_size=batch_size, mode='test', src_lang='ko', tgt_lang='en', preload_voca=True) # load parallel corpus
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

saver = tf.train.Saver()
# init session vars
model_names = ['logs/model/big/model-414.ckpt', './finetuned_model-0.ckpt', './finetuned_model_all-0.ckpt']

opt_loss = 10000
opt_idx = 0

def validation(model_idx):
    global opt_loss, opt_idx

    # run graph for translating
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    model_name = model_names[model_idx]
    print('model: ', model_name)
    saver.restore(sess, 'logs/model/big/' + model_name)

    t_avg = 1.
    t_timer = time.time()

    l_avg = 0.0
    for j in range(data.num_batch):
        t_str = time.time()
        # predict character
        out = sess.run(loss, 
                              {x: data.source[j*batch_size:(j+1)*batch_size], 
                               y: data.target[j*batch_size:(j+1)*batch_size], 
                               y0: np.zeros((batch_size,1)),
                               p_keep_conv: 0.9})
        t_elp = time.time() - t_str
        t_avg = 0.9*t_avg + 0.1*t_elp
        l_avg += np.mean(out) / data.num_batch
        t_rem = t_avg * (data.num_batch-j-1)
        print('[%d: %.2fs, %dm]' % (j, t_elp, t_rem//60), np.mean(out), end='\r')
    print('\n' + str(l_avg) +'\n')

    if l_avg < opt_loss:
        opt_loss = l_avg
        opt_idx = model_idx
    

    
## Search phase 1
#for i in range(len(model_names)//10):
#    validation(i*10)
#validation(-1)

#opt_idx = 33
#
#print(opt_idx, model_names[opt_idx])

## Search phase 2
#sentinel_idx = opt_idx
#for i in range(11):
#    if i == 5:
#        continue
#
#    idx = sentinel_idx + i - 5
#    if 0 <= idx < len(model_names):
#        validation(idx)


search_list = [27,28,29,34,35,37,39,42,49,51,52,54,55,56,57]
for i in range(len(search_list)):
    validation(search_list[i])

print(opt_idx, model_names[opt_idx])


