import time
import numpy as np
import tensorflow as tf
from data_koen import KOEN
from bytenet import ByteNet

# set log level to debug
tf.logging.set_verbosity(tf.logging.DEBUG)


#
# hyper parameters
#
batch_size = 32   # batch size
latent_dim = 200   # hidden layer dimension
num_block_sets = 4     # dilated blocks
filter_size = 3
learning_rate = 0.0003

data = KOEN(batch_size, 'train') # load parallel corpus
model = ByteNet(data.voca_size, data.max_len, latent_dim, num_block_sets)

# vanilla loss
# cross entropy loss with logit and mask 
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model.y, logits=model.logits)
loss = tf.identity(loss)
loss *= tf.cast(tf.not_equal(model.y, tf.zeros_like(model.y)), loss.dtype)
loss = tf.reduce_mean(tf.reduce_sum(loss, 1))

saver = tf.train.Saver(max_to_keep=20)

# run graph for translating
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# init session vars
checkpoint_dir = 'logs/models/'
latest_model_path = tf.train.latest_checkpoint(checkpoint_dir)
cnt = 0

if latest_model_path:
    print('model exsists: ', latest_model_path)
    saver.restore(sess, latest_model_path)

    import re
    cnt = int(re.search('model-(\w+).ckpt', latest_model_path).group(1))

# optimizer
temp = set(tf.global_variables())
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
uninit_vars = set(tf.global_variables()) - temp
sess.run(tf.variables_initializer(uninit_vars))

# save and test
def save_and_test():
    global cnt
    cnt += 1
    save_path = saver.save(sess, (checkpoint_dir + "model-%d.ckpt") % (cnt))
    test_size = 10 
    # initialize character sequence
    pred_prev = np.zeros((test_size, data.max_len)).astype(np.int32)
    pred = np.zeros((test_size, data.max_len)).astype(np.int32)

    # output of encoder
    x_enc = sess.run(model.enc, {model.x: data.source[:test_size]})

    # generate output sequence
    for i in range(data.max_len):
        # predict character
        out = sess.run(model.label, {model.enc: x_enc, model.y_src: pred_prev,
                              model.p_keep_conv: 1.0})
        # update character sequence
        if i < data.max_len - 1:
            pred_prev[:, i+1] = out[:, i]
        pred[:, i] = out[:, i]
    # print result
    print('\nsources : --------------')
    data.print_index(data.source[:test_size])
    print('\ntargets : --------------')
    data.print_index(pred[:,:])

t_avg = 1.
num_iter = 32
l_avg = 1.
t_timer = time.time()
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
                              {model.x: data.source[j*batch_size:(j+1)*batch_size], 
                               model.y: data.target[j*batch_size:(j+1)*batch_size], 
                               model.y0: np.zeros((batch_size,1)),
                               model.p_keep_conv: 0.9})
        t_elp = time.time() - t_str
        t_avg = 0.9*t_avg + 0.1*t_elp
        l_avg = 0.99*l_avg + 0.01*np.mean(out2)
        t_rem = t_avg * ((num_iter-i-1)*(data.num_batch) + (data.num_batch-j-1))
        print('[%d, %d: %.2fs, %dm]' % (i, j, t_elp, t_rem//60), l_avg, end='\r')

        if time.time() > t_timer + 3600:
            t_timer = time.time()
            print()
            save_and_test()

    print()
    save_and_test()
