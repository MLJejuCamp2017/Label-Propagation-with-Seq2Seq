
# coding: utf-8

# In[1]:

from tqdm import tqdm


# In[2]:

import tensorflow as tf
import numpy as np


# # ByteNet

# In[4]:

def layer_norm(input, causal=False, name=None):
    '''
    Layer Normalization
    
    If the model is causal and using Convnet,
    normalize input only according to depth.
    '''
    with tf.variable_scope('layer_norm', name):
        if causal: # Sub Layer Normalization
            axis_depth = len(input.get_shape()) - 1
            mean, var = tf.nn.moments(input, [axis_depth], keep_dims=True)
            out = (input - mean) / tf.sqrt(var)
            return out
        else: # Layer Normalization
            axes = np.arange(len(input.get_shape()) - 1) + 1
            mean, var = tf.nn.moments(input, axes, keep_dims=True)
            out = (input - mean) / tf.sqrt(var)
            return out

def convolution(input, filter, padding, strides=None, dilation_rate=None, causal=False, name=None):
    '''
    Masked Convolution
    
    See PixelCNN
    '''
    with tf.variable_scope('masked_convolution', name):
        filter_shape = filter.get_shape().as_list()
        filter_len = np.prod(filter_shape[:-2])
        center = filter_len // 2
        if causal:
            mask = np.ones([filter_len] + filter_shape[-2:], dtype='float32')
            mask[center+1: ,: ,:] = 0.
            mask = mask.reshape(filter_shape)
            
            mask = tf.constant(mask, dtype='float32')
            filter = filter * mask


        ret = tf.nn.convolution(input, filter, padding=padding, strides=strides,
                                dilation_rate=dilation_rate, name=name)
        
    return ret


def res_block(input, filter_size=3, dilation_rate=None, causal=False, name='res_block'):
    '''
    Residual block
    
    For details, see Ch3.6(Fig 3. Left) of 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    '''
    
    with tf.variable_scope(name):
        x = input
        
        # input dimension
        in_dim = input.get_shape().as_list()[-1]
    
        # normalization
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # reduce dimension
        w_shape = [1, in_dim, in_dim//2]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev),
                            name='w1')
        x = tf.nn.convolution(x, w, padding='SAME')
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # 1xk conv dilated (with mask)
        w_shape = [filter_size, in_dim//2, in_dim//2]
        if causal:
            w_stddev = np.sqrt(2. / (np.prod(w_shape[1:-1]) * (filter_size//2 + 1)))
        else:
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev),
                            name='w2')
        x = convolution(x, w, padding='SAME', dilation_rate=dilation_rate, causal=causal)
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # dimension recover and residual connection
        w_shape = [1, in_dim//2, in_dim]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev),
                            name='w3')
        x = tf.nn.convolution(x, w, padding='SAME')
        
        # residual connection
        x = x + input
        

    return x


def encoder(input, filter_size=3, num_block_sets=6):
    '''
    Encoder for Character-Level Machine Translation
    
    For details, see Ch6 of 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    '''
    with tf.variable_scope('encoder'):
        x = input
        for i in range(num_block_sets):
            for j in [1,2,4,8,16]:
                x = res_block(x, filter_size=filter_size, dilation_rate=[j], name='res_block_%d_%d' % (i, j))
        
    return x

def decoder(input, filter_size=3, num_block_sets=6):
    '''
    Decoder for Character-Level Machine Translation
    
    For details, see Ch6 of 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    '''
    with tf.variable_scope('decoder'):
        x = input
        for i in range(num_block_sets):
            for j in [1,2,4,8,16]:
                x = res_block(x, filter_size=filter_size, dilation_rate=[j],
                              causal=True, name='res_block_%d_%d' % (i, j))
        
    return x


# In[3]:

class ByteNet(object):
    """
    ByteNet

    For details, see 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    """ 

    def __init__(self, input_dim=254, input_max_len=150, latent_dim=200, num_block_sets=4):
        self.input_dim = input_dim
        self.input_max_len = input_max_len
        self.filter_size = 3

        self.latent_dim = latent_dim
        self.num_block_sets = num_block_sets

    def encoder(self, x):
        #
        # inputs
        #
        with tf.variable_scope('input'):
            # make embedding matrix for source and target
            emb_x = tf.get_variable(shape=[self.input_dim, self.latent_dim],
                                    initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                    name='emb_x')

        #
        # encode graph ( atrous convolution )
        #

        # embed table lookup
        enc_emb = tf.nn.embedding_lookup(emb_x, x)
        enc = encoder(enc_emb, filter_size=self.filter_size, num_block_sets=self.num_block_sets)

        return enc

    def decoder(self, enc, y, p_keep_conv):
        #
        # inputs
        #
        with tf.variable_scope('input'):
            emb_y = tf.get_variable(shape=[self.input_dim, self.latent_dim], 
                                    initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                    name='emb_y')
            y_src = tf.pad(y[:,:-1], [[0,0],[1,0]])

        #
        # decode graph ( causal convolution )
        #

        # loop dilated causal conv block
        dec_emb = tf.concat([enc, tf.nn.embedding_lookup(emb_y, y_src)], 2)
        dec = decoder(dec_emb, filter_size=self.filter_size, num_block_sets=self.num_block_sets)


        with tf.variable_scope('output'):
            # additional convolution and relu
            out = layer_norm(dec, causal=True)
            out = tf.nn.relu(out)
            out_dim = out.get_shape().as_list()[-1] # latent_dim * 2
            w_shape = [1, out_dim, out_dim]
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
            w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev),
                                name='w1')
            out = tf.nn.convolution(out, w, padding='SAME')

            # final fully convolution layer for softmax
            logits = layer_norm(out, causal=True)
            logits = tf.nn.relu(logits)

            logits = tf.nn.dropout(logits, p_keep_conv)

            w_shape = [1, out_dim, self.input_dim]
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
            w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev),
                                name='w2')
            logits = tf.nn.convolution(logits, w, padding='SAME')

        return logits


# ## Test
# 
# ### Baseline

# from preprocess import MAX_LEN

# ### NGM (Neural Graph Model)

# In[5]:

from preprocess import MAX_LEN
from batch import batch_iter


# In[6]:

# hyperparameters
latent_dim = 100   # hidden layer dimension
num_block_sets = 2     # dilated blocks


# In[7]:

def tower_loss1(model, params):
    p_keep_conv, alpha1, alpha2, alpha3, in_u1, in_v1, in_u2,    in_v2, in_u3, in_v3, labels_u1, labels_v1,    labels_u2, weights_ll, weights_lu, weights_uu,    cu1, cv1, cu2, labels_zero_1, labels_zero_2, labels_zero_3    = params
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        enc_u1 = model.encoder(in_u1)
        logits_u1 = model.decoder(enc_u1, labels_u1, p_keep_conv)
        
        scope.reuse_variables()
        
        enc_v1 = model.encoder(in_v1)
        enc_u2 = model.encoder(in_u2)

        logits_v1 = model.decoder(enc_v1, labels_v1, p_keep_conv)
        logits_u2 = model.decoder(enc_u2, labels_u2, p_keep_conv)
        
    loss_function = tf.reduce_mean(cu1 * vanilla_loss(logits_u1, labels_u1))                    + tf.reduce_mean(cv1 * vanilla_loss(logits_v1, labels_v1))                    + tf.reduce_mean(cu2 * vanilla_loss(logits_u2, labels_u2))
            
    return loss_function, enc_u1, enc_v1, enc_u2


# In[8]:

def tower_loss2(model, params, enc_u1, enc_v1, enc_u2):
    p_keep_conv, alpha1, alpha2, alpha3, in_u1, in_v1, in_u2,    in_v2, in_u3, in_v3, labels_u1, labels_v1,    labels_u2, weights_ll, weights_lu, weights_uu,    cu1, cv1, cu2, labels_zero_1, labels_zero_2, labels_zero_3    = params
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        
        scope.reuse_variables()
        
        enc_v2 = model.encoder(in_v2)
        enc_u3 = model.encoder(in_u3)
        enc_v3 = model.encoder(in_v3)
        
        scores_u1 = enc_u1
        scores_v1 = enc_v1
        scores_u2 = enc_u2
        scores_v2 = enc_v2
        scores_u3 = enc_u3
        scores_v3 = enc_v3
        #scores_u1 = model.decoder(enc_u1, labels_zero_1, p_keep_conv)
        #scores_v1 = model.decoder(enc_v1, labels_zero_1, p_keep_conv)
        #scores_u2 = model.decoder(enc_u2, labels_zero_2, p_keep_conv)
        #scores_v2 = model.decoder(enc_v2, labels_zero_2, p_keep_conv)
        #scores_u3 = model.decoder(enc_u3, labels_zero_3, p_keep_conv)
        #scores_v3 = model.decoder(enc_v3, labels_zero_3, p_keep_conv)
    
    loss_function = tf.reduce_mean(alpha1 * weights_ll * distance_loss(scores_u1, scores_v1))                    + tf.reduce_mean(alpha2 * weights_lu * distance_loss(scores_u2, scores_v2))                    + tf.reduce_mean(alpha3 * weights_uu * distance_loss(scores_u3, scores_v3))
            
    return loss_function


# In[9]:

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# In[10]:

def make_params():
    p_keep_conv = tf.placeholder(tf.float32, [])

    alpha1 = tf.constant(0.10, dtype=np.float32, name="a1")
    alpha2 = tf.constant(0.10, dtype=np.float32, name="a2")
    alpha3 = tf.constant(0.05, dtype=np.float32, name="a3")
    in_u1 = tf.placeholder(tf.int32, [None, MAX_LEN], name="ull")
    in_v1 = tf.placeholder(tf.int32, [None, MAX_LEN], name="vll")
    in_u2 = tf.placeholder(tf.int32, [None, MAX_LEN], name="ulu")
    in_v2 = tf.placeholder(tf.int32, [None, MAX_LEN], name="vlu")
    in_u3 = tf.placeholder(tf.int32, [None, MAX_LEN], name="ulu")
    in_v3 = tf.placeholder(tf.int32, [None, MAX_LEN], name="ulu")
    labels_u1 = tf.placeholder(tf.int32, [None, MAX_LEN], name="lu1")
    labels_v1 = tf.placeholder(tf.int32, [None, MAX_LEN], name="lv1")
    labels_u2 = tf.placeholder(tf.int32, [None, MAX_LEN], name="lu2")
    weights_ll = tf.placeholder(tf.float32, [None, ], name="wll")
    weights_lu = tf.placeholder(tf.float32, [None, ], name="wlu")
    weights_uu = tf.placeholder(tf.float32, [None, ], name="wuu")
    cu1 = tf.placeholder(tf.float32, [None, ], name="CuLL")
    cv1 = tf.placeholder(tf.float32, [None, ], name="CvLL")
    cu2 = tf.placeholder(tf.float32, [None, ], name="CuLU")

    labels_zero_1 = tf.placeholder(tf.int32, [None, MAX_LEN], name="l0_1")
    labels_zero_2 = tf.placeholder(tf.int32, [None, MAX_LEN], name="l0_2")
    labels_zero_3 = tf.placeholder(tf.int32, [None, MAX_LEN], name="l0_3")
    
    return [p_keep_conv, alpha1, alpha2, alpha3, in_u1, in_v1, in_u2,    in_v2, in_u3, in_v3, labels_u1, labels_v1,    labels_u2, weights_ll, weights_lu, weights_uu,    cu1, cv1, cu2, labels_zero_1, labels_zero_2, labels_zero_3]


# In[11]:

# vanilla loss
# cross entropy loss with logit and mask 
def vanilla_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.identity(loss)
    loss *= tf.cast(tf.not_equal(labels, tf.zeros_like(labels)), loss.dtype)
    loss = tf.reduce_sum(loss, 1)

    return loss


# In[12]:

# distance loss
def distance_loss(scores_u, scores_v):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u, labels=tf.nn.softmax(scores_v))
    loss = tf.reduce_sum(loss, 1)
    return loss


# In[13]:

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

num_gpus = len(get_available_gpus())


# In[14]:

with tf.Graph().as_default(), tf.device('/cpu:0'):

    model = ByteNet(latent_dim=latent_dim, num_block_sets=num_block_sets)
    optimizer = tf.train.AdamOptimizer(1e-3)
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        params = make_params()
        with tf.device('/gpu:0'):

            loss1, enc_u1, enc_v1, enc_u2 = tower_loss1(model, params)
            scope.reuse_variables()
 
        with tf.device('/gpu:1'):
            
            loss2 = tower_loss2(model, params, enc_u1, enc_v1, enc_u2)

        #with tf.device('/gpu:2'):
            loss_function = loss1 + loss2
    

    train = optimizer.minimize(loss_function)
    init = tf.global_variables_initializer()
    
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    saver = tf.train.Saver(var_list=sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))


# In[43]:
"""
num_epochs = 1

[p_keep_conv, alpha1, alpha2, alpha3, in_u1, in_v1, in_u2,    in_v2, in_u3, in_v3, labels_u1, labels_v1,    labels_u2, weights_ll, weights_lu, weights_uu,    cu1, cv1, cu2, labels_zero_1, labels_zero_2, labels_zero_3] = params

data_ko2en.print_index(data_ko2en.target[:batch_size]) 
for epoch in range(num_epochs):
    print("======== EPOCH " + str(epoch + 1) + " ========")

    batches = batch_iter(batch_size=128)
    epoch_loss = 0
    
    cnt = 0
    for batch in tqdm(batches):

        u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu = batch
        
        l0_1 = np.zeros(u1.shape)
        l0_2 = np.zeros(u2.shape)
        l0_3 = np.zeros(u3.shape)
        _, loss = sess.run([train, loss_function],
                                feed_dict={in_u1: u1,
                                           in_v1: v1,
                                           in_u2: u2,
                                           in_v2: v2,
                                           in_u3: u3,
                                           in_v3: v3,
                                           labels_u1: lu1,
                                           labels_v1: lv1,
                                           labels_u2: lu2,
                                           weights_ll: w_ll,
                                           weights_lu: w_lu,
                                           weights_uu: w_uu,
                                           cu1: c_ull,
                                           cv1: c_vll,
                                           cu2: c_ulu,
                                           p_keep_conv: 0.9,
                                           labels_zero_1: l0_1,
                                           labels_zero_2: l0_2,
                                           labels_zero_3: l0_3})
        epoch_loss += loss
        print(loss, end='\r')
        cnt += 1
        
        if(cnt % 5 == 0):
            print()
            test_func()
            saver.save(sess, './logs/models/ngm-model-%d.ckpt' % cnt, write_meta_graph=False)
        
    print()
    print("Epoch_Loss", epoch_loss/cnt)
"""

# In[15]:



import time
from data import KOEN

# set log level to debug
tf.logging.set_verbosity(tf.logging.DEBUG)

batch_size = 8  # batch size
latent_dim = 100
num_blocks = 2 # for encoder, decoder
num_layers = 5 # 1, 2, 4, 8, 16

##
## TEST DATA SET
##
data_ko2en = KOEN(batch_size=batch_size, mode='valid')

input_dim = data_ko2en.voca_size
input_max_len = data_ko2en.max_len

graph = sess.graph


# In[20]:

with graph.as_default(), tf.device('/gpu:1'):
    x = tf.placeholder(tf.int32, [None, MAX_LEN], name='x')
    y = tf.placeholder(tf.int32, [None, MAX_LEN], name='y')
    p = tf.constant(1.0, dtype=tf.float32, name='p')
    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        enc_x = model.encoder(x)
        logits_y = model.decoder(enc_x, y, p)


# In[21]:

##
## SET WEIGHTS
##
train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
var_names = [x.name for x in train_vars]

weights = []
for b in range(num_blocks):
    weights.append([])
    for i in range(num_layers):
        weights[b].append([])
        rate = 2**i
        
        w_names = 'decoder/res_block_%d_%d/w' % (b, rate)
        w_names = [w_names+x+':0' for x in ['1', '2', '3']]
        
        for w_name in w_names:
            weights[b][i].append(train_vars[var_names.index(w_name)])

weights.append([])
weights[-1].append(train_vars[var_names.index('output/w1:0')])
weights[-1].append(train_vars[var_names.index('output/w2:0')])


# In[22]:

def simple_norm(x):
    mean, var = tf.nn.moments(x, [1], keep_dims=True)
    
    return (x - mean) / tf.sqrt(var + 1e-5)


def linear_block(w, inputs, state):
    x = inputs
    x = tf.matmul(state, w[0]) + tf.matmul(x, w[1])
    x = simple_norm(x)
    x = tf.nn.relu(x)
    
    return x

def linear_output(variables, inputs):
    x = inputs
    x = simple_norm(x)
    x = tf.nn.relu(x)

    w0 = variables[0][0]
    x = tf.matmul(x, w0)
    x = simple_norm(x)
    x = tf.nn.relu(x)

    w1 = variables[1][0]
    x = tf.matmul(x, w1)

    return x


def linear_init(sess):
    with sess.graph.as_default():
        input_size = latent_dim * 2
        inputs = tf.placeholder(tf.float32, [batch_size, input_size], name='inputs')
        
        h = inputs
        # block op
        init_ops = []
        push_ops = []
        pull_ops = []
        qs = []
        dummies = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                
                x = h
                x = simple_norm(x)
                x = tf.nn.relu(x)
                x = tf.matmul(x, weights[b][i][0][0])
                x = simple_norm(x)
                x = tf.nn.relu(x)
                
                # fast op
                state_size = latent_dim
                    
                q = tf.FIFOQueue(rate,
                                 dtypes=tf.float32,
                                 shapes=(batch_size, state_size))
                dummy = tf.zeros((rate, batch_size, state_size))
                init = q.enqueue_many(dummy)
                pull_op = q.dequeue_many(rate)
            
                
                state_ = q.dequeue()
                push = q.enqueue([x])
                init_ops.append(init)
                push_ops.append(push)
                pull_ops.append(pull_op)
                qs.append(q)
                dummies.append(dummy)
                
                # block op
                x = linear_block(weights[b][i][1], x, state_)
                x = tf.matmul(x, weights[b][i][2][0])

                h = x + h

        outputs = linear_output(weights[-1], h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)
    
    return inputs, init_ops, out_ops, pull_ops, qs, dummies


# In[24]:

def run(sess, batch_src):
    
    predictions = []
    batch = np.zeros(len(batch_src), 'int32')
    x_enc = sess.run(enc_x, {x: batch_src})
    
    for step in range(input_max_len):
        # make batch
        batch = np.concatenate((x_enc[:,step], emb_y[batch]), axis=1)
        feed_dict = {inputs: batch}
        output = sess.run(out_ops, feed_dict=feed_dict)[0] # ignore push ops
        
        batch = np.argmax(output,1)
        predictions.append(batch)

    predictions_ = np.array(predictions).T
    return predictions_

def clean(sess):
            
    all_ops = []
    all_ops.extend(pull_ops)
    all_ops.extend(init_ops)
    sess.run(all_ops)


# In[25]:

inputs, init_ops, out_ops, pull_ops, qs, dummies = linear_init(sess)
# Initialize queues.
sess.run(init_ops)
rates = sess.run([q.size() for q in qs])


# In[26]:

def copy_qs(sess, ids):
    '''
    Copy queue values of two elements in a batch
    '''
   
    qs_vals = sess.run(pull_ops)
    
    dummies_dict = {}
    for i in range(len(qs)):
        q = qs[i]

        q_val = qs_vals[i]
        q_val = q_val[:, ids]
        
        dummies_dict[dummies[i]] = q_val

    sess.run(init_ops, dummies_dict)
    
def beam_run(sess, batch_src, batch_size=4, beam_size=8):
    # Beam Search Variables
    end_flags = np.zeros([batch_size*beam_size],dtype='int32')
    storage_c = np.zeros([batch_size*beam_size, input_max_len], dtype='int32')
    storage_p = np.zeros([batch_size*beam_size, input_max_len], dtype='float32')
    loglike = np.zeros([batch_size*beam_size], dtype='float32')
    
    predictions = []
    batch = np.zeros(batch_size*beam_size, 'int32')
    x_enc = sess.run(enc, {x: batch_src})
    x_enc = x_enc[np.array([[i] * beam_size for i in range(len(x_enc))]).flatten()] # upsampling
    
    for step in range(input_max_len):
        # make batch
        batch = np.concatenate((x_enc[:,step], emb_y[batch]), axis=1)
        feed_dict = {inputs: batch}
        
        output = sess.run(out_ops, feed_dict=feed_dict)[0] # ignore push ops
        
        # Beam search
        M_p = output # M_p : unnormalized probabilities. [batch_size*beam_size, input_dim]
        M_p = M_p / np.sum(M_p, axis=1).reshape(-1,1) # normalize probabilities
        
        # block ended seqs
        block_ids = np.where(end_flags)[0]

        M_p[M_p<1e-45] = 1e-45
        # calculate log_likelihoods
        M_l = loglike.reshape(-1,1) + (1-end_flags).reshape(-1,1)*np.log(M_p)
        
        if step==0: # remove all except first values for similarity breaking
            mask = np.array([[1] + [0]*(beam_size-1)]*batch_size).flatten()
            M_l = M_l * mask.reshape(-1,1)
            M_l += np.array([0, -np.inf])[(1-mask)].reshape(-1,1)

        # calculate scores
        len_y = np.argmin(np.not_equal(storage_c,0), axis=1) + 1
        len_penalty = (((1 + len_y) / (1 + 5)) ** 0.65)
        M_s = M_l / len_penalty.reshape(-1,1)

        # find indices of top-n scores
        M_s_flatten = M_s.reshape(-1, input_dim*beam_size)
        col_ids = np.flip(np.argsort(M_s_flatten, axis=1), axis=1)[:,:beam_size]
        col_ids = col_ids.reshape(-1)
        row_ids = np.array([[i*beam_size]*beam_size for i in range(batch_size)]).flatten()
        row_ids += col_ids // input_dim
        col_ids = col_ids % input_dim

        # update variables
        eos = 1

        end_flags = end_flags[row_ids]

        storage_c = storage_c[row_ids]
        storage_c[:,step] = col_ids * (1-end_flags)

        storage_p = storage_p[row_ids]
        storage_p[:,step] = M_p[row_ids, col_ids] * (1-end_flags)

        loglike = M_l[row_ids, col_ids] * (1-end_flags) + loglike * end_flags 

        end_flags = (col_ids==eos) * (1-end_flags) + end_flags
        
        batch = storage_c[:,step]
        if((batch==0).all()):
            break
        
        copy_qs(sess, row_ids)
        
    predictions_ = storage_c[np.arange(batch_size) * beam_size]
    return predictions_


# In[38]:

def test_func():
    global emb_y 
    
    t_avg = 10.
    ret = []
    emb_y = sess.run(train_vars[var_names.index('input/emb_y:0')])
    for i in range(1): #data_ko2en.num_batch * beam_size):
        t_str = time.time()

        predictions = run(sess, data_ko2en.source[i*batch_size:(i+1)*batch_size])
        clean(sess)
        ret.extend(data_ko2en.print_index(predictions, sysout=False))

        t_elp = time.time() - t_str
        t_avg = 0.9*t_avg + 0.1*t_elp
        t_rem = t_avg * (data_ko2en.num_batch-i-1)
        print('[%d: %.2fs, %dm]' % (i, t_elp, t_rem//60), t_avg)
    
    data_ko2en.print_index(predictions)


# In[ ]:

num_epochs = 1

[p_keep_conv, alpha1, alpha2, alpha3, in_u1, in_v1, in_u2,    in_v2, in_u3, in_v3, labels_u1, labels_v1,    labels_u2, weights_ll, weights_lu, weights_uu,    cu1, cv1, cu2, labels_zero_1, labels_zero_2, labels_zero_3] = params

data_ko2en.print_index(data_ko2en.target[:batch_size]) 
for epoch in range(num_epochs):
    print("======== EPOCH " + str(epoch + 1) + " ========")

    batches = batch_iter(batch_size=128)
    epoch_loss = 0
    
    cnt = 0
    for batch in tqdm(batches):

        u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu = batch
        
        l0_1 = np.zeros(u1.shape)
        l0_2 = np.zeros(u2.shape)
        l0_3 = np.zeros(u3.shape)
        _, loss = sess.run([train, loss_function],
                                feed_dict={in_u1: u1,
                                           in_v1: v1,
                                           in_u2: u2,
                                           in_v2: v2,
                                           in_u3: u3,
                                           in_v3: v3,
                                           labels_u1: lu1,
                                           labels_v1: lv1,
                                           labels_u2: lu2,
                                           weights_ll: w_ll,
                                           weights_lu: w_lu,
                                           weights_uu: w_uu,
                                           cu1: c_ull,
                                           cv1: c_vll,
                                           cu2: c_ulu,
                                           p_keep_conv: 0.9,
                                           labels_zero_1: l0_1,
                                           labels_zero_2: l0_2,
                                           labels_zero_3: l0_3})
        epoch_loss += loss
        print(loss, end='\r')
        cnt += 1
        
        if(cnt % 450 == 0):
            print()
            test_func()
            saver.save(sess, './logs/models/ngm-model-%d.ckpt' % cnt, write_meta_graph=False)
        
    print()
    print("Epoch_Loss", epoch_loss/cnt)

