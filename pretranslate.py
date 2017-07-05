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

# greedy search policy
label = tf.argmax(logits, 2)

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

#
# translate
#

# smaple french sentences for source language
sources = [
    "내년 내 가장 큰 목표는 억대 연봉자가 되는 거야.",
    "손을 엉덩이 위에 올리세요.",
    "그는 이 회사의 대주주다",
    "루이스의 일원(Norman Baker)은 정말 의문을 제기하고 있었다.",
    "그녀는 감정에 충실하다.",
    "오늘밤 우리가 여기서 뭐 하나라도 해결했나?",
    "총격 소리는 전쟁지역에서 멈추지 않는다.",
    "백업된 폴더 및 하위 폴더를 추가 또는 제거합니다.",
    "누르면 안내가 나옵니다.",
    "우리 아빠는 매일 하는 일에 지치셨다.",
    "이 점은 강조할 필요가 없다.",
    "우리는 죽음을 두려워하지 않아요.",
    "저거 안먹어도 돼",
    "나는 새 차를 매입할 것 같다.",
    "블라우스 하나만 더 입어 보구요.",
    "세상이 어떻게 돌아가고 있는거야?",
]

import unicodedata
sources = [unicodedata.normalize("NFKD", s) for s in sources]

# to batch form
sources = data.to_batch(sources, 'en')

test_size = len(sources) 

# initialize character sequence
pred_prev = np.zeros((test_size, data.max_len)).astype(np.int32)
pred = np.zeros((test_size, data.max_len)).astype(np.int32)

# output of encoder
x_enc = sess.run(enc, {x: sources})

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
data.print_index(sources, has_token=True)
print('\ntargets : --------------')
data.print_index(pred[:,:])
print()
