import time
import numpy as np
import tensorflow as tf
from batch import batch_iter
from ngm_bytenet import g

# set log level to debug
tf.logging.set_verbosity(tf.logging.DEBUG)


#
# hyper parameters
#
batch_size = 32   # batch size
latent_dim = 100   # hidden layer dimension
num_block_sets = 2     # dilated blocks
filter_size = 3
learning_rate = 0.0003
max_len = 150

y0 = tf.zeros(dtype='int32', shape=[batch_size, 1])
p_keep_conv = tf.placeholder(tf.float32)

alpha1 = tf.constant(0.10, dtype=np.float32, name="a1")
alpha2 = tf.constant(0.10, dtype=np.float32, name="a2")
alpha3 = tf.constant(0.05, dtype=np.float32, name="a3")
in_u1 = tf.placeholder(tf.int32, [None, max_len], name="ull")
in_v1 = tf.placeholder(tf.int32, [None, max_len], name="vll")
in_u2 = tf.placeholder(tf.int32, [None, max_len], name="ulu")
in_v2 = tf.placeholder(tf.int32, [None, max_len], name="vlu")
in_u3 = tf.placeholder(tf.int32, [None, max_len], name="ulu")
in_v3 = tf.placeholder(tf.int32, [None, max_len], name="ulu")
labels_u1 = tf.placeholder(tf.float32, [None, max_len], name="lu1")
labels_v1 = tf.placeholder(tf.float32, [None, max_len], name="lv1")
labels_u2 = tf.placeholder(tf.float32, [None, max_len], name="lu2")
labels_v2 = tf.placeholder(tf.float32, [None, max_len], name="lv2")
labels_u3 = tf.placeholder(tf.float32, [None, max_len], name="lu3")
labels_v3 = tf.placeholder(tf.float32, [None, max_len], name="lv3")
weights_ll = tf.placeholder(tf.float32, [None, ], name="wll")
weights_lu = tf.placeholder(tf.float32, [None, ], name="wlu")
weights_uu = tf.placeholder(tf.float32, [None, ], name="wuu")
cu1 = tf.placeholder(tf.float32, [None, ], name="CuLL")
cv1 = tf.placeholder(tf.float32, [None, ], name="CvLL")
cu2 = tf.placeholder(tf.float32, [None, ], name="CuLU")
test_input = tf.placeholder(tf.int32, [None, len_input, ], name="test_input")
test_labels = tf.placeholder(tf.float32, [None, 2], name="test_labels")

with tf.variable_scope("ngm") as scope:
    scores_u1 = g(in_u1, labels_u1, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scope.reuse_variables()
    scores_v1 = g(in_v1, labels_v1, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scores_u2 = g(in_u2, labels_u2, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scores_v2 = g(in_v2, labels_v2, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scores_u3 = g(in_u3, labels_u3, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scores_v3 = g(in_v3, labels_v3, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)
    scores_test = g(test_input, test_labels, y0, p_keep_conv, data.voca_size, data.max_len, latent_dim, num_block_sets)

# vanilla loss
# cross entropy loss with logit and mask 
def vanilla_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.identity(loss)
    loss *= tf.cast(tf.not_equal(labels, tf.zeros_like(labels)), loss.dtype)
    loss = tf.reduce_mean(tf.reduce_sum(loss, 1))

    return loss

loss_function = tf.reduce_mean(alpha1 * weights_ll * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u1, labels=tf.nn.softmax(scores_v1)) \
                            + cu1 * vanilla_loss(logits=scores_u1, labels=labels_u1) \
                            + cv1 * vanilla_loss(logits=scores_v1, labels=labels_v1)) \
                            + tf.reduce_mean(alpha2 * weights_lu * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u2, labels=tf.nn.softmax(scores_v2)) \
                            + cu2 * vanilla_loss(logits=scores_u2, labels=labels_u2)) \
                            + tf.reduce_mean(alpha3 * weights_uu * tf.nn.softmax_cross_entropy_with_logits(logits=scores_u3, labels=tf.nn.softmax(scores_v3)))

saver = tf.train.Saver(max_to_keep=20)

# run graph for translating
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# init session vars
checkpoint_dir = 'logs/ngm_models/'
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
train = optimizer.minimize(loss_function)
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
    x_enc = sess.run(enc, {model.x: data.source[:test_size]})

    # generate output sequence
    for i in range(data.max_len):
        # predict character
        out = sess.run(scores_test, {model.enc: x_enc, model.y_src: pred_prev,
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
