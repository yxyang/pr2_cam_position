import numpy as np, scipy.io, tensorflow as tf, matplotlib.pyplot as plt

data = scipy.io.loadmat('./gazebo_data.mat')
X, y = data['images'], data['labels']
X1 = np.zeros([X.shape[0], X.shape[1] * X.shape[2]])
for i in range(X.shape[0]):
    X1[i,:] = X[i,:,:].flatten()
image_height = X.shape[1]
image_width = X.shape[2]

X = X1
n = y.shape[0]
image_size = image_height * image_width
labels_count = y.shape[1]

print("Input data has been loaded...")

num_train = n * 4 // 5; num_cv = n - num_train
perm = np.random.permutation(n)
Xtrain = X[perm[:num_train], :]
ytrain = y[perm[:num_train], :]
Xcv = X[perm[num_train:], :]
ycv = y[perm[num_train:], :]

print("And parsed...")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Probably need to edit this method to speed things up
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder('float32', shape = [None, image_size])
y_ = tf.placeholder('float32', shape = [None, labels_count])

#20 x 20 patches, 1 input channel, 32 output units
w_conv1 = weight_variable([20, 20, 1, 32])
# bias variable (output of the 1st layer as well)
b_conv1 = bias_variable([32])

#Reshape to 2d image
image = tf.reshape(X, [-1, image_height , image_width, 1])
h_conv1 = tf.nn.relu(conv2d(image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# First Layer ^ convolution + pooling

# Reshape the 32 lin. comb. of inputs of 2nd layer to 4x8 grid for visualization(?)
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))  
layer1 = tf.transpose(layer1, (0, 3, 1, 4, 2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8)) 

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
layer2 = tf.reshape(h_conv2, (-1, image_height//2, image_width//2, 4 ,16))  
layer2 = tf.transpose(layer2, (0, 3, 1, 4, 2))
layer2 = tf.reshape(layer2, (-1, image_height//2*4, image_width//2*16)) 

W_fc1 = weight_variable([80*60 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 80*60*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float32')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

mean_squared = tf.reduce_sum(tf.square(tf.sub(y, y_)))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
loss = cross_entropy
LEARNING_RATE = 1e-4
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

epoches_completed = 0
index_in_epoch = 0
n = Xtrain.shape[0]
def nextBatch(batch_size):
    global Xtrain
    global ytrain
    global index_in_epoch
    global epoches_completed
    
    if (index_in_epoch + batch_size >= n):
        epoches_completed += 1
        index_in_epoch = 0
        perm = np.arange(n)
        np.random.shuffle(perm)
        Xtrain = Xtrain[perm, :]
        ytrain = ytrain[perm, :]
    return Xtrain[index_in_epoch:index_in_epoch + batch_size, :], ytrain[index_in_epoch:index_in_epoch + batch_size, :]

print("Starting session...")
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

print("Running whatever you coded...")
print(loss.eval(feed_dict={X:Xtrain, y_: ytrain, keep_prob: 1.0}))