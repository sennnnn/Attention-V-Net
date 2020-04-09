import tensorflow as tf
import tensorflow.layers as layers

# layer parameters
DECAY_BATCH_NORM = 0.9
EPSILON = 1E-05
LEAKY_RELU = 0.1

def CB(input, filters, strides=1, kernel_size=3):
    """
    convolution + batch normalization
    """
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def CBR(input, filters=None, strides=1, kernel_size=3):
    """
    convolution + batch normalization + leaky relu operation
    """
    if(filters == None):
        filters = input.get_shape().as_list()[-1]
    input = layers.conv2d(input, filters, kernel_size, use_bias=True ,strides=strides, padding='same', \
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def ACBR(input, filters, rate, kernel_size=3):
    """
    atrous convolution + batch normalization
    """
    c = input.get_shape().as_list()[-1]
    filters_variable = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, c, filters], dtype=tf.float32))
    input = tf.nn.atrous_conv2d(input, filters_variable, rate, padding='SAME')
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)
    input = tf.nn.leaky_relu(input, alpha=LEAKY_RELU, name='ac')
    
    return input

def upsampling(input, filters, kernel_size=3, strides=2):
    """
    convolution_transpose + batch normalization
    """
    # Up-sampling Layer,implemented by transpose convolution.
    input = layers.conv2d_transpose(input, filters, kernel_size, strides, padding='same')
    input = layers.batch_normalization(input, momentum=DECAY_BATCH_NORM, epsilon=EPSILON)

    return input

def meta_block(input, filters, keep_prob=None):
    input = CBR(input, filters)
    if(keep_prob != None):
        input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = CBR(input, filters)

    return input

def r_block(input, filters, t):
    for i in range(t):

        if i == 0:
            add = CBR(input, filters)
        
        input = CBR(add + input, filters)

    return input

def r2_block(input, filters, t):
    input = CBR(input, filters, kernel_size=1)
    raw = input
    input = r_block(input, filters, t)
    input = r_block(input, filters, t)

    return input + raw

def DAC(input):
    c = input.get_shape().as_list()[-1]
    
    sub1 = ACBR(input, c, 1)
    sub1 = ACBR(sub1, c, 3)
    sub1 = ACBR(sub1, c, 5)
    sub1 = ACBR(sub1, c, 1, 1)

    sub2 = ACBR(input, c, 1)
    sub2 = ACBR(sub2, c, 3)
    sub2 = ACBR(sub2, c, 1, 1)

    sub3 = ACBR(input, c, 3)
    sub3 = ACBR(input, c, 1, 1)

    sub4 = ACBR(input, c, 1)

    return input+sub1+sub2+sub3+sub4

def RMC(input):

    sub1 = layers.max_pooling2d(input, 2, strides=2, padding='same');sub1 = CBR(sub1, 1, kernel_size=1);sub1 = upsampling(sub1, 1)
    sub2 = layers.max_pooling2d(input, 3, strides=2, padding='same');sub2 = CBR(sub2, 1, kernel_size=1);sub2 = upsampling(sub2, 1)
    sub3 = layers.max_pooling2d(input, 5, strides=2, padding='same');sub3 = CBR(sub3, 1, kernel_size=1);sub3 = upsampling(sub3, 1)
    sub4 = layers.max_pooling2d(input, 6, strides=2, padding='same');sub4 = CBR(sub4, 1, kernel_size=1);sub4 = upsampling(sub4, 1)

    out = tf.concat([sub4, sub3, sub2, sub1, input], axis=-1)

    return out

def attention_gate_block(input_g, input_l, f_in):
    input_g = CB(input_g, f_in)
    out = input_l
    input_l = CB(input_l, f_in)
    fuse = tf.nn.relu(input_g+input_l)
    fuse = CB(fuse, 1, kernel_size=1)
    fuse = tf.nn.sigmoid(fuse)

    return out*fuse

def _unet_(input, num_class, initial_channel=64, keep_prob=0.5, degree=4):
    c = initial_channel
    o = input
    fuse_list = []

    for i in range(degree):
        o = CBR(o, c)
        o = CBR(o, c)
        fuse_list.append(o)
        o = CBR(o, c, strides=2)
        c = c*2

    o = CBR(o, c)
    o = tf.nn.dropout(o, keep_prob=keep_prob)
    o = CBR(o, c)

    for i in range(degree)[::-1]:
        c = c//2
        o = upsampling(o, c)
        o = tf.concat([fuse_list[i], o], axis=-1)
        o = CBR(o, c)
        o = CBR(o, c)
    
    o = CBR(o, num_class, kernel_size=1)

    return o

## ============ networks ============ ##

def unet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel
    o = input
    fuse_list = []

    for i in range(4):
        o = CBR(o, c)
        o = CBR(o, c)
        fuse_list.append(o)
        o = CBR(o, c, strides=2)
        c = c*2

    o = CBR(o, c)
    o = tf.nn.dropout(o, keep_prob=keep_prob)
    o = CBR(o, c)

    for i in range(4)[::-1]:
        c = c//2
        o = upsampling(o, c)
        o = tf.concat([fuse_list[i], o], axis=-1)
        o = CBR(o, c)
        o = CBR(o, c)
    
    o = CBR(o, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def vnet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel
    o = input
    
    o = CBR(o, c)
    res_o = o
    o = CBR(o, c)
    o = res_o + o
    fuse1 = o
    c = c*2
    o = CBR(o, c, strides=2)

    res_o = o
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o
    fuse2 = o
    c = c*2
    o = CBR(o, c, strides=2)

    res_o = o
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o
    fuse3 = o
    c = c*2
    o = CBR(o, c, strides=2)

    res_o = o
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o
    fuse4 = o
    c = c*2
    o = CBR(o, c, strides=2)
    
    o = CBR(o, c)
    o = tf.nn.dropout(o, keep_prob=keep_prob)
    o = CBR(o, c)

    c = c//2
    o = upsampling(o, c)
    res_o = o
    o = tf.concat([fuse4, o], axis=-1)
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o

    c = c//2
    o = upsampling(o, c)
    res_o = o
    o = tf.concat([fuse3, o], axis=-1)
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o

    c = c//2
    o = upsampling(o, c)
    res_o = o
    o = tf.concat([fuse2, o], axis=-1)
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o

    c = c//2
    o = upsampling(o, c)
    res_o = o
    o = tf.concat([fuse1, o], axis=-1)
    o = CBR(o, c)
    o = CBR(o, c)
    o = res_o + o

    o = CBR(o, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def r2unet(input, num_class, initial_channel=64, keep_prob=0.5, t=2):
    c = initial_channel
    
    input = r2_block(input, c, t)
    fus1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = r2_block(input, c, t)
    fus2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = r2_block(input, c, t)
    fus3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = r2_block(input, c, t)
    fus4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = tf.nn.dropout(input, keep_prob=keep_prob)
    input = r2_block(input, c, t)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fus4, input], axis=-1)
    input = r2_block(input, c, t)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fus3, input], axis=-1)
    input = r2_block(input, c, t)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fus2, input], axis=-1)
    input = r2_block(input, c, t)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fus1, input], axis=-1)
    input = r2_block(input, c, t)

    o = CBR(input, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def unetpp(input, num_class, initial_channel=64, keep_prob=0.5):
    """
    Unet++ network architecture.
    """
    c = initial_channel;x0_0 = input
    c1 = c;c2 = c*2;c3 = c*4;c4 = c*8;c5 = c*16
    x0_0 = meta_block(x0_0, c1)
    x1_0 = CBR(x0_0, c1, strides=2);x1_0 = meta_block(x1_0, c2, keep_prob)
    x0_1 = meta_block(tf.concat([x0_0, upsampling(x1_0, c2)], axis=-1), c1)

    x2_0 = CBR(x1_0, c2, strides=2);x2_0 = meta_block(x2_0, c3, keep_prob)
    x1_1 = meta_block(tf.concat([x1_0, upsampling(x2_0, c3)], axis=-1), c2)
    x0_2 = meta_block(tf.concat([x0_0, x0_1, upsampling(x1_1, c2)], axis=-1), c1)

    x3_0 = CBR(x2_0, c3, strides=2);x3_0 = meta_block(x3_0, c4, keep_prob)
    x2_1 = meta_block(tf.concat([x2_0, upsampling(x3_0, c4)], axis=-1), c3)
    x1_2 = meta_block(tf.concat([x1_0, x1_1, upsampling(x2_1, c3)], axis=-1), c2)
    x0_3 = meta_block(tf.concat([x0_0, x0_1, x0_2, upsampling(x1_2, c2)], axis=-1), c1)

    x4_0 = CBR(x3_0, c4, strides=2);x4_0 = meta_block(x4_0, c5, keep_prob)
    x3_1 = meta_block(tf.concat([x3_0, upsampling(x4_0, c5)], axis=-1), c4)
    x2_2 = meta_block(tf.concat([x2_0, x2_1, upsampling(x3_1, c4)], axis=-1), c3)
    x1_3 = meta_block(tf.concat([x1_0, x1_1, x1_2, upsampling(x2_2, c3)], axis=-1), c2)
    x0_4 = meta_block(tf.concat([x0_0, x0_1, x0_2, x0_3, upsampling(x1_3, c2)], axis=-1), c1)

    o = CBR(x0_4, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def cenet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel
    input = CBR(input, c)
    input = CBR(input, c)
    fuse1 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse2 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse3 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = CBR(input, c)
    input = CBR(input, c)
    fuse4 = input
    input = CBR(input, c, strides=2)

    c = c*2
    input = DAC(input)
    input = tf.nn.dropout(input, keep_prob)
    input = RMC(input)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse4, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse3, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse2, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    c = c//2
    input = upsampling(input, c)
    input = tf.concat([fuse1, input], axis=-1)
    input = CBR(input, c)
    input = CBR(input, c)

    o = CBR(input, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def attention_unet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel

    fuse_list = []

    for _ in range(4):
        input = CBR(input, c)
        input = CBR(input, c)
        fuse_list.append(input)
        input = CBR(input, c, strides=2)
        c = c*2

    input = CBR(input, c)
    input = tf.nn.dropout(input, keep_prob)
    input = CBR(input, c)

    for index in range(4)[::-1]:
        c = c//2
        input = upsampling(input, c)
        fuse = attention_gate_block(input, fuse_list[index], c//2)
        input = tf.concat([fuse, input], axis=-1)
        input = CBR(input, c)
        input = CBR(input, c)

    o = CBR(input, num_class, kernel_size=1)
    o = tf.identity(o, name='output')

    return o

def mnet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel
    o = input
    pass

def attention_guided_net(input, num_class, initial_channel=64, keep_prob=0.5):
    pass

def multires_unet(input, num_class, initial_channel=64, keep_prob=0.5):
    pass

def zigzag_unet(input, num_class, initial_channel=64, keep_prob=0.5):
    c = initial_channel
    o = input
    out_list = []

    with tf.variable_scope('zigzag_1'):
        o = _unet_(o, num_class, c, keep_prob, 4)
        o = tf.identity(o, name='output')
        out_list.append(o)

    with tf.variable_scope('zigzag_2'):
        o = _unet_(o, num_class, c, keep_prob, 3)
        o = tf.identity(o, name='output')
        out_list.append(o)

    with tf.variable_scope('zigzag_3'):
        o = _unet_(o, num_class, c, keep_prob, 2)
        o = tf.identity(o, name='output')
        out_list.append(o)

    with tf.variable_scope('zigzag_4'):
        o = _unet_(o, num_class, c, keep_prob, 1)
        o = tf.identity(o, name='output')
        out_list.append(o)

    return tuple(out_list)

    
