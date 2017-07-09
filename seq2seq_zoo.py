
# coding: utf-8

# In[1]:

from tqdm import tqdm


# In[2]:

import tensorflow as tf
import numpy as np


# # ByteNet

# In[3]:

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


# In[4]:

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

# In[5]:

from preprocess import MAX_LEN
from batch import batch_iter


# In[6]:

# hyperparameters
latent_dim = 100   # hidden layer dimension
num_block_sets = 2     # dilated blocks


# In[7]:

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


# In[8]:

with tf.variable_scope('model') as scope:
    model = ByteNet(latent_dim=latent_dim, num_block_sets=num_block_sets)
    enc_u1 = model.encoder(in_u1)
    logits_u1 = model.decoder(enc_u1, labels_u1, p_keep_conv)


# In[9]:

with tf.variable_scope('model', reuse=True) as scope:
    enc_v1 = model.encoder(in_v1)
    enc_u2 = model.encoder(in_u2)
    enc_v2 = model.encoder(in_v2)
    enc_u3 = model.encoder(in_u3)
    enc_v3 = model.encoder(in_v3)    


# In[10]:

with tf.variable_scope('model', reuse=True) as scope:
    logits_v1 = model.decoder(enc_v1, labels_v1, p_keep_conv)
    logits_u2 = model.decoder(enc_u2, labels_u2, p_keep_conv)
    
    scores_u1 = model.decoder(enc_u1, labels_zero_1, p_keep_conv)
    scores_v1 = model.decoder(enc_v1, labels_zero_1, p_keep_conv)
    scores_u2 = model.decoder(enc_u2, labels_zero_2, p_keep_conv)
    scores_v2 = model.decoder(enc_v2, labels_zero_2, p_keep_conv)
    scores_u3 = model.decoder(enc_u3, labels_zero_3, p_keep_conv)
    scores_v3 = model.decoder(enc_v3, labels_zero_3, p_keep_conv)


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

loss_function = tf.reduce_mean(cu1 * vanilla_loss(logits_u1, labels_u1))                    + tf.reduce_mean(cv1 * vanilla_loss(logits_v1, labels_v1))                    + tf.reduce_mean(cu2 * vanilla_loss(logits_u2, labels_u2))


# In[ ]:

loss_function += tf.reduce_mean(alpha1 * weights_ll * distance_loss(scores_u1, scores_v1))                    + tf.reduce_mean(alpha2 * weights_lu * distance_loss(scores_u2, scores_v2))                    + tf.reduce_mean(alpha3 * weights_uu * distance_loss(scores_u3, scores_v3))


# In[ ]:

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss_function)


# In[ ]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:

num_epochs = 1
for epoch in range(num_epochs):
    print("======== EPOCH " + str(epoch + 1) + " ========")

    batches = batch_iter(batch_size=3)
    epoch_loss = 0
    
    cnt = 0
    for batch in tqdm(batches):

        u1, v1, lu1, lv1, u3, v3, u2, v2, lu2, w_ll, w_lu, w_uu, c_ull, c_vll, c_ulu = batch
        
        l0_1 = np.zeros(u1.shape)
        l0_2 = np.zeros(u2.shape)
        l0_3 = np.zeros(u3.shape)
        _, loss = sess.run([optimizer, loss_function],
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
        print(cnt, loss, end='\r')
        cnt += 1
        
    print()
    print("Epoch_Loss", epoch_loss/cnt)

