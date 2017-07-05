# coding: utf-8

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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
args = parser.parse_args()

batch_size = args.batch_size  # batch size

##
## TEST DATA SET
##
data = KOEN(batch_size=batch_size, mode='test', src_lang='ko', tgt_lang='en', preload_voca=True)

input_dim = data.voca_size
input_max_len = data.max_len


##
## RESTORE PRETRAINED GRAPH
##
graph = tf.Graph()
with graph.as_default():
    importer = tf.train.import_meta_graph('test_models/model-414.ckpt.meta')

sess = tf.Session(graph=graph)
importer.restore(sess, 'test_models/finetuned_model_all-0.ckpt')

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
## INITIALIZE VARIABLES
##

var_names = [x.name for x in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
uninitialized_vars = sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[var_names.index('beta1_power:0'):]

init_new_vars_op = tf.variables_initializer(uninitialized_vars)

sess.run(init_new_vars_op)

##
## VALIDATION
##

def test():
    for i in range(1):#data.num_batch):
        # initialize character sequence
        pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
        pred = np.zeros((batch_size, data.max_len)).astype(np.int32)

        # output of encoder
        x_enc = sess.run(enc, feed_dict={x: data.source[i*batch_size:(i+1)*batch_size]})

        for j in range(data.max_len):
            # predict character
            t_str = time.time()
            out = sess.run(label, feed_dict={enc: x_enc, y_src: pred_prev,
                                                            p_keep_conv: 1.0})

            # update character sequence
            if j < data.max_len - 1:
                    pred_prev[:, j+1] = out[:, j]
            pred[:, j] = out[:, j]

            t_elp = time.time() - t_str
            print('[%d: %.2fs]' % (i, t_elp), end='\n')
        # print result
        #print('\nsources : --------------')
        #data.print_index(data.source[i*batch_size:(i+1)*batch_size], has_token=True)
        #print('\ntruths : --------------')
        #data.print_index(data.target[i*batch_size:(i+1)*batch_size])

        #print('\ntargets : --------------')
        #data.print_index(pred[:,:])
  
test()
