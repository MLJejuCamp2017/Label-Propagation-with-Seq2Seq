import tensorflow as tf
import numpy as np

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
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev))
        x = tf.nn.convolution(x, w, padding='SAME')
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # 1xk conv dilated (with mask)
        w_shape = [filter_size, in_dim//2, in_dim//2]
        if causal:
            w_stddev = np.sqrt(2. / (np.prod(w_shape[1:-1]) * (filter_size//2 + 1)))
        else:
            w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev))
        x = convolution(x, w, padding='SAME', dilation_rate=dilation_rate, causal=causal)
        x = layer_norm(x, causal)
        x = tf.nn.relu(x)
        
        # dimension recover and residual connection
        w_shape = [1, in_dim//2, in_dim]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev))
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


def g_e(x, input_dim, input_max_len, latent_dim,  num_block_sets):
    """
    ByteNet - Encoder

    For details, see 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    """ 

    filter_size = 3

    #
    # inputs
    #
    with tf.variable_scope('input'):
        # make embedding matrix for source and target
        emb_x = tf.get_variable(shape=[input_dim, latent_dim], initializer=tf.random_uniform_initializer(-1.0, 1.0))

    #
    # encode graph ( atrous convolution )
    #

    # embed table lookup
    enc_emb = tf.nn.embedding_lookup(emb_x, x)
    enc = encoder(enc_emb, filter_size=filter_size, num_block_sets=num_block_sets)
        
    return enc
                            
def g_d(enc, y, y0, p_keep_conv, input_dim, input_max_len, latent_dim,  num_block_sets):
    """
    ByteNet - Decoder

    For details, see 'Neural Machine Translation in Linear Time(https://arxiv.org/abs/1610.10099)'.
    """ 

    filter_size = 3

    #
    # inputs
    #
    with tf.variable_scope('input'):
        emb_y = tf.get_variable(shape=[input_dim, latent_dim], initializer=tf.random_uniform_initializer(-1.0, 1.0))\
        # shift target for training source
        y_src = tf.concat([y0, y[:, :-1]], 1)

    #
    # decode graph ( causal convolution )
    #

    # loop dilated causal conv block
    dec_emb = tf.concat([enc, tf.nn.embedding_lookup(emb_y, y_src)], 2)
    dec = decoder(dec_emb, filter_size=filter_size, num_block_sets=num_block_sets)


    with tf.variable_scope('output'):
        # additional convolution and relu
        out = layer_norm(dec, causal=True)
        out = tf.nn.relu(out)
        out_dim = out.get_shape().as_list()[-1] # latent_dim * 2
        w_shape = [1, out_dim, out_dim]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev))
        out = tf.nn.convolution(out, w, padding='SAME')

        # final fully convolution layer for softmax
        logits = layer_norm(out, causal=True)
        logits = tf.nn.relu(logits)

        p_keep_conv = tf.placeholder(tf.float32) # should be 0.9
        logits = tf.nn.dropout(logits, p_keep_conv)

        w_shape = [1, out_dim, input_dim]
        w_stddev = np.sqrt(2./np.prod(w_shape[:-1])) # He's init
        w = tf.get_variable(shape=w_shape, initializer=tf.random_normal_initializer(stddev=w_stddev))
        logits = tf.nn.convolution(logits, w, padding='SAME')
        
    return logits

