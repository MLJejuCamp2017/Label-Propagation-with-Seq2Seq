import tensorflow as tf
import numpy as np

def layer_norm(input, causal=False, name=None):
    '''
    Layer Normalization
    
    If the model is causal and using Convnet,
    normalize input only according to depth.
    '''
    with tf.name_scope('layer_norm', name):
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
    with tf.name_scope('masked_convolution', name):
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
    
    with tf.name_scope(name):
        x = input
        
        # input dimension
        in_dim = input.get_shape().as_list()[-1]
    
        # normalization
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # reduce dimension
        w_shape = [1, in_dim, in_dim//2]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
        x = tf.nn.convolution(x, w, padding='SAME')
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # 1xk conv dilated (with mask)
        w_shape = [filter_size, in_dim//2, in_dim//2]
        if causal:
            w_stddev = np.sqrt(2. / (np.prod(w_shape[1:-1]) * (filter_size//2 + 1)))
        else:
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
        x = convolution(x, w, padding='SAME', dilation_rate=dilation_rate, causal=causal)
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # dimension recover and residual connection
        w_shape = [1, in_dim//2, in_dim]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
        x = tf.nn.convolution(x, w, padding='SAME')
        
        # residual connection
        x = x + input
        

    return x


def encoder(input, filter_size=3, num_block_sets=6):
    '''
    Encoder for Character-Level Machine Translation
    
    For details, see Ch6 of 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    '''
    with tf.name_scope('encoder'):
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
    with tf.name_scope('decoder'):
        x = input
        for i in range(num_block_sets):
            for j in [1,2,4,8,16]:
                x = res_block(x, filter_size=filter_size, dilation_rate=[j],
                              causal=True, name='res_block_%d_%d' % (i, j))
        
    return x


class ByteNet(object):
    """
    ByteNet

    For details, see 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    """ 

    def __init__(self, input_dim, input_max_len, latent_dim=200, num_block_sets=4):
        self.input_dim = input_dim
        self.input_max_len = input_max_len
        self.filter_size = 3

        self.latent_dim = latent_dim
        self.num_block_sets = num_block_sets

        self._build()

    def _build(self):
        #
        # inputs
        #
        with tf.name_scope('input'):
            # place holders
            x = tf.placeholder(dtype='int32', shape=[None, self.input_max_len])
            y = tf.placeholder(dtype='int32', shape=[None, self.input_max_len])

            # make embedding matrix for source and target
            emb_x = tf.Variable(tf.random_uniform([self.input_dim, self.latent_dim], -1.0, 1.0))
            emb_y = tf.Variable(tf.random_uniform([self.input_dim, self.latent_dim], -1.0, 1.0))

            # shift target for training source
            y0 = tf.placeholder(dtype='int32', shape=[None, 1])
            y_src = tf.concat([y0, y[:, :-1]], 1)

            self.x, self.y, self.emb_x, self.emb_y, self.y0, self.y_src\
                    = x, y, emb_x, emb_y, y0, y_src

        #
        # encode graph ( atrous convolution )
        #

        # embed table lookup
        enc_emb = tf.nn.embedding_lookup(emb_x, x)
        enc = encoder(enc_emb, filter_size=self.filter_size, num_block_sets=self.num_block_sets)

        #
        # decode graph ( causal convolution )
        #

        # loop dilated causal conv block
        dec_emb = tf.concat([enc, tf.nn.embedding_lookup(emb_y, y_src)], 2)
        dec = decoder(dec_emb, filter_size=self.filter_size, num_block_sets=self.num_block_sets)

        self.enc_emb, self.enc, self.dec_emb, self.dec\
                = enc_emb, enc, dec_emb, dec

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

            w_shape = [1, out_dim, self.input_dim]
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
            w = tf.Variable(tf.random_normal(w_shape, stddev=w_stddev))
            logits = tf.nn.convolution(logits, w, padding='SAME')
            
        # greedy search policy
        label = tf.argmax(logits, 2) 

        self.p_keep_conv, self.logits, self.label\
                = p_keep_conv, logits, label 

