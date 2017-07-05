# coding: utf-8

import time
import numpy as np
import tensorflow as tf

import os
import sys
from data_koen import KOEN

# set log level to debug
tf.logging.set_verbosity(tf.logging.DEBUG)


#
# hyper parameters
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
args = parser.parse_args()

batch_size = args.batch_size  # batch size
latent_dim = 400
num_blocks = 6 # for encoder, decoder
num_layers = 5 # 1, 2, 4, 8, 16

##
## TEST DATA SET
##
data_ko2en = KOEN(batch_size=batch_size, mode='test', src_lang='ko', tgt_lang='en', preload_voca=True)

input_dim = data_ko2en.voca_size
input_max_len = data_ko2en.max_len

##
## RESTORE PRETRAINED GRAPH
##
graph = tf.Graph()
with graph.as_default():
    importer = tf.train.import_meta_graph('test_models/model-414.ckpt.meta')

sess = tf.Session(graph=graph)
model_name = 'model-414.ckpt'
importer.restore(sess, 'test_models/%s' % model_name)

##
## SET VARIABLES
##
x = sess.graph.get_operation_by_name('input/Placeholder').outputs[0]
enc = sess.graph.get_operation_by_name('encoder/res_block_5_16/add').outputs[0]
y = sess.graph.get_operation_by_name('input/Placeholder_1').outputs[0]
y0 = sess.graph.get_operation_by_name('input/Placeholder_2').outputs[0]
y_src = sess.graph.get_operation_by_name('input/concat').outputs[0]
p_keep_conv = sess.graph.get_operation_by_name('output/Placeholder').outputs[0]


logits = sess.graph.get_operation_by_name('output/convolution_1/Squeeze').outputs[0] # logits
loss = sess.graph.get_operation_by_name('Mean').outputs[0] # loss
label = sess.graph.get_operation_by_name('ArgMax').outputs[0] # label

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
        
        w_names = 'decoder/res_block_%d_%d/Variable' % (b, rate)
        w_names = [w_names+x+':0' for x in ['', '_1', '_2']]
        
        for w_name in w_names:
            weights[b][i].append(train_vars[var_names.index(w_name)])

weights.append([])
weights[-1].append(train_vars[var_names.index('output/Variable:0')])
weights[-1].append(train_vars[var_names.index('output/Variable_1:0')])


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

emb_y = sess.run(train_vars[var_names.index('input/Variable:0')])

def run(sess, batch_src):
    
    predictions = []
    batch = np.zeros(len(batch_src), 'int32')
    x_enc = sess.run(enc, {x: batch_src})
    
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

inputs, init_ops, out_ops, pull_ops, qs, dummies = linear_init(sess)
# Initialize queues.
sess.run(init_ops)
rates = sess.run([q.size() for q in qs])

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

t_avg = 10.
ret = []
beam_size = 12
for i in range(1): #data_ko2en.num_batch * beam_size):
    t_str = time.time()

    predictions = run(sess, data_ko2en.source[i*batch_size:(i+1)*batch_size])
    clean(sess)
    ret.extend(data_ko2en.print_index(predictions, sysout=False))

    t_elp = time.time() - t_str
    t_avg = 0.9*t_avg + 0.1*t_elp
    t_rem = t_avg * (data_ko2en.num_batch-i-1)
    print('[%d: %.2fs, %dm]' % (i, t_elp, t_rem//60), t_avg)
