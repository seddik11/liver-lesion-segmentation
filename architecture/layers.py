import numpy as np
import tensorflow as tf


# Variables

def weight_variable(shape, stddev=0.1, name=None):
    """
    Creates a weight variable initialized with a truncated normal distribution.

    Parameters
    ----------
    shape: list or tuple of ints
        The shape of the weight variable.
    stddev: float
        The standard deviation of the truncated normal distribution.
    name : string
        The name of the variable in TensorFlow.

    Returns
    -------
    weights: TF variable
        The weight variable.   
    """
    
    # return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())

    initializer=tf.contrib.layers.variance_scaling_initializer()
    return tf.Variable(initializer(shape), name=name, dtype=tf.float32)

    # return tf.Variable(initial_value=tf.truncated_normal(shape, stddev=stddev),
    #                    name=name,
    #                    dtype=tf.float32)

def weight_fixed(shape, name=None):
    
    return tf.Variable(initial_value=tf.constant(1.0,shape=shape),
                       name=name,
                       dtype=tf.float32)


def bias_variable(shape, name=None):
    """
    Creates a bias variable initialized with a constant value.

    Parameters
    ----------
    shape: list or tuple of ints
        The shape of the bias variable.
    name : string
        The name of the variable in TensorFlow.

    Returns
    -------
    bias: TF variable
        The bias variable.   
    """
    return tf.Variable(initial_value=tf.constant(1., shape=shape),
                       name=name,
                       dtype=tf.float32)


# Layers and Operations

def upconv2d(x, w, stride, output_shape, data_format="NHWC"):
    """
    A simple up-convolution without activation function.

    Parameters
    ----------
    x: TF tensor
        The layer input.
    w : TF variable
        The weights.
    output_shape: TF tensor
        The shape of the output.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    upconv2d: TF operation
        The (output of the) 2D upconvolution.   
    """

    if data_format == "NCHW":
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 1, stride, stride], padding='VALID', data_format=data_format, name='upconv')

    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='VALID', data_format=data_format, name='upconv')


def max_pool(x, n, data_format="NHWC"):
    """
    A simple max pooling layer.

    Parameters
    ----------
    x: TF tensor
        The layer input.
    n : int
        The size of the max pooling region.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    max_pool: TF operation
        The (output of the) 2D max pooling layer.     
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    if data_format == "NCHW":
        return tf.nn.max_pool(x, ksize=[1, 1, n, n], strides=[1, 1, n, n], padding='VALID', data_format=data_format, name='max_pool')
    
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', data_format=data_format, name='max_pool')


def dropout(x, keep_prob):
    """
    A simple drop-out layer.

    Parameters
    ----------
    x: TF tensor
        The layer input.
    keep_prob: float
        The keep probability.

    Returns
    -------
    dropout: TF operation
        The (output of the) dropout layer.     
    """
    return tf.nn.dropout(x, keep_prob, name='dropout')

def conv(x, out_filters=None, filter_size=3, stride=1, channel_multiplier=1, padding="SAME", name="conv"):

    shape = x.shape.as_list()
    in_filters = shape[3]
    stddev = np.sqrt(2. / in_filters)
    if out_filters is None:
        out_filters = in_filters * channel_multiplier
        if type(out_filters) in [float, np.float32, np.float64]:
            out_filters = int(round(out_filters))
    with tf.name_scope(name):
        w = weight_variable([filter_size, filter_size, in_filters, out_filters], stddev, name='weights')
        b = bias_variable([1, 1, out_filters], name='biases')
        cnv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding, data_format="NHWC", name='conv')
        out = tf.nn.relu(cnv + b, name="relu")
    return out

def trans(x, stride=2, shape=None, out_filters=12, name="trans", data_format="NHWC"):

    n = stride
    in_filters = x.shape.as_list()[3]
    stddev = np.sqrt(2. / in_filters)
    with tf.name_scope(name):
        w = weight_variable([n, n, out_filters, in_filters], stddev, name='weights')
        b = bias_variable([1, 1, out_filters], name='biases')
        upconv = tf.nn.conv2d_transpose(x, w, shape, strides=[1, n, n, 1], padding='SAME', data_format=data_format, name='conv')
        upconv = tf.nn.relu(upconv + b, name='relu')

    return upconv


def atrous(y, filter_size=3, out_filters=None, dilation_rate=1, padding="SAME", batch_norm=False, training=False, dense=False, outputs=None, merge=None):

    x = y
    if outputs is not None:
        if merge is not None:
            # x = tf.concat([x, outputs[merge]], 3)
            x = outputs[merge]
    shape = x.shape.as_list()
    in_filters = shape[3]

    stddev = np.sqrt(2. / (filter_size**2 * in_filters))
    with tf.name_scope('atrous'):
        out = x
        if batch_norm:
            out = tf.layers.batch_normalization(out, axis=3, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)
            out = tf.nn.relu(out, name='relu')
        w = weight_variable([filter_size, filter_size, in_filters, out_filters], stddev, name='weights')
        #layer_weights = tf.summary.histogram('layer' + str(index) + '_weights', w, collections=['measurements'])
        b = bias_variable([1, 1, out_filters], name='biases')
        #layer_weights = tf.summary.histogram('layer' + str(index) + '_biases', b, collections=['measurements'])
        conv = tf.nn.atrous_conv2d(out, w, dilation_rate, padding=padding, name='atrous_conv2d')
        out = conv + b
        if not batch_norm:
            out = tf.nn.relu(out, name='relu')
        #layer_activations = tf.summary.histogram('layer' + str(index) + '_activations', out, collections=['activations'])
        if outputs is not None:
            if merge is None:
                outputs.append(out)
        if dense:
            out = tf.concat([y, out], 3)

    return out

def convpool(x, filter_size, channel_multiplier=1, padding="SAME", pool_size=2, name="convpool", data_format="NHWC"):
    
    n = pool_size
    with tf.name_scope(name):
        x = tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', data_format=data_format, name='max_pool')
        out = conv(x, filter_size=3, stride=1, channel_multiplier=2, padding="SAME")
    return out

def transconv(x, shape=None, filter_size=3, stride=1, channel_multiplier=1, padding="SAME", name="transconv"):
    
    with tf.name_scope(name):
        x = trans(x, stride=stride, shape=shape, out_filters=shape[3])
        out = conv(x, filter_size=filter_size, stride=1)
    return out

def spread_down(layer, filter_size=3, out_filters=12, dilation_rate=1, pool_size=2, padding="SAME", batch_norm=False, training=False, dense=False):
    
    x, y = layer
    with tf.name_scope('spread_down'):
        x = atrous(x, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, padding="SAME", batch_norm=False, training=False, dense=dense)
        y = convpool(y, filter_size=filter_size, channel_multiplier=2, padding=padding, pool_size=pool_size)
    
    return (x, y)

def spread_up(layer, filter_size=3, out_filters=12, dilation_rate=1, padding="SAME", batch_norm=False, training=False, dense=False, concats=None):
    
    x, y = layer
    with tf.name_scope('spread_up'):
       
        x = atrous(x, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, padding="SAME", dense=dense)
        y = transconv(y, shape=concats[1].shape.as_list(), filter_size=filter_size, padding=padding, stride=2)
        
        if concats is not None:
            x = tf.concat([x, concats[0]], 3, name='concat1')
            y = tf.concat([y, concats[1]], 3, name='concat2')
            x = atrous(x, filter_size=filter_size, out_filters=out_filters, dilation_rate=1, padding="SAME", dense=dense)
            y = conv(y, filter_size=filter_size, channel_multiplier=0.5, stride=1, padding="SAME")
    
    return (x, y)

def ffru_down(layer, index=0, pool_size=2, filter_size=3):
    
    # full resolution, downsampled layers
    x, y = layer
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()

    with tf.name_scope('ffru_down'):

        # calculate the appropriate stride
        n = 1
        for i in xrange(index):
            n *= pool_size

        # downsample the full resolution layer by learning features
        # down_x = conv(x, out_filters=y_shape[3], filter_size=n, stride=n, padding="SAME", name="conv1")
        down_x = tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', data_format="NHWC", name='max_pool')

        # concatenate the downsampled full resolution layer with the downsampled layer
        z = tf.concat([down_x, y], 3, name='concat1')

        # conv layer for the concatenated layers
        z = conv(z, filter_size=filter_size, stride=1, padding="SAME", name="conv2")

        # upsample to the full resolution
        upconv = trans(z, stride=n, shape=x_shape, out_filters=x_shape[3])

        # summarize informations for the downsampled layer
        downconv = conv(z, out_filters=y_shape[3], filter_size=filter_size, stride=1, padding="SAME", name="conv3")
        
        # concatentation of the exchanged informations with the originals
        x = tf.concat([x, upconv], 3, name='concat2')
        y = tf.concat([y, downconv], 3, name='concat3')

        # fusion the informations
        x = conv(x, filter_size=filter_size, out_filters=x_shape[3], stride=1, padding="SAME", name="conv4")
        y = conv(y, out_filters=y_shape[3], filter_size=filter_size, stride=1, padding="SAME", name="conv5")

    return (x, y)

def ffru_up(layer, filter_size=3, data_format="NHWC"):
    
    # full resolution, downsampled layers
    x, y = layer
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()

    with tf.name_scope('ffru_up'):

        # calculate the appropriate stride
        n = x_shape[1] / y_shape[1]

        # upample the downsampled layer
        up_y = trans(y, n, shape=x_shape, out_filters=x_shape[3])

        # concatenate the upsample of the downsampled layer with the full resolution layer
        z = tf.concat([x, up_y], 3, name='concat1')

        # conv layer for the concatenated layers
        z = conv(z, filter_size=filter_size, stride=1, padding="SAME", name="conv1")

        # summarize information for the full resolution layer
        upconv = conv(z, out_filters=x_shape[3], filter_size=filter_size, stride=1, padding="SAME", name="conv2")

        # downsample to the downsampled layer
        downconv = tf.nn.max_pool(z, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID', data_format=data_format, name='max_pool')
        
        # concatentation of the exchanged informations with the originals
        x = tf.concat([x, upconv], 3, name='concat2')
        y = tf.concat([y, downconv], 3, name='concat3')

        # fusion the informations
        x = conv(x, out_filters=x_shape[3], filter_size=filter_size, stride=1, padding="SAME", name="conv3")
        y = conv(y, out_filters=y_shape[3], filter_size=filter_size, stride=1, padding="SAME", name="conv4")
        
    return (x, y)



# Blocks

def conv_block(x, filter_size, out_filters=None, channel_multiplier=1, convolutions=2, padding="VALID", data_format="NHWC"):
    """
    A block of convolutions with ReLU activations.

    Parameters
    ----------
    x: TF tensor
        The layer input.
    filter_size: int
        The side length of the filters.
    out_filters: int
        The depth of the output.
    channel_multiplier: int or float
        Alternative way to compute the output depth as a multiple of the input depth.
    convolutions: int
        The number of convolutional layers.
    padding: string
        Either "SAME" for padding or "VALID" for no padding.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    conv_block: TF operation
        The output of the convolutional block.     
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    #if data_format == "NHWC":
    #   raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    shape = x.shape.as_list()
    in_filters = shape[3]
    if out_filters is None:
        out_filters = channel_multiplier * in_filters
        if type(out_filters) in [float, np.float32, np.float64]:
            out_filters = int(round(out_filters))

    stddev = np.sqrt(2. / (filter_size**2 * in_filters))

    with tf.name_scope('conv1'):
        w = weight_variable([filter_size, filter_size, in_filters, out_filters], stddev, name='weights')
        b = bias_variable([1,1,out_filters], name='biases')

        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding, data_format=data_format, name='conv')
        current_tensor = tf.nn.relu(conv + b, name='relu')

    stddev = np.sqrt(2. / (filter_size**2 * out_filters))

    for i in xrange(convolutions - 1):
        with tf.name_scope('conv' + str(i + 2)):
            w = weight_variable([filter_size, filter_size, out_filters, out_filters], stddev, name='weights')
            b = bias_variable([1,1,out_filters], name='biases')

            conv = tf.nn.conv2d(current_tensor, w, strides=[1, 1, 1, 1], padding=padding, data_format=data_format, name='conv')
            current_tensor = tf.nn.relu(conv + b, name='relu')

    return current_tensor

def res_block(x, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=False, activation="relu", batch_norm=True, data_format="NHWC"):
    """
    A block of convolutions with ReLU activations.

    Parameters
    ----------
    x: TF tensor
        The layer input.
    filter_size: int
        The side length of the filters.
    channel_multiplier: int or float
        Alternative way to compute the output depth as a multiple of the input depth.
    depthwise_multiplier: int
        The depthwise multiplier for the depthwise convolution of the depthwise separable convolutions.
    convolutions: int
        The number of convolutional layers.
    training: bool or TF bool
        The boolean value, which switches batch normalization to training or inference mode.
    activation: string
        Either "ReLU" for the standard ReLU activation or "cReLU" for the concatenated ReLU activation function.
    batch_norm: bool
        Whether to use batch normalization or not.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    conv_block: TF operation
        The output of the convolutional block.     
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    #if data_format == "NHWC":
    #    raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    activation = str.lower(activation)
    if activation not in ["relu", "crelu"]:
        raise ValueError("activation must be \"ReLU\" or \"cReLU\".")

    shape = x.shape.as_list()
    in_filters = shape[1]
    out_filters = channel_multiplier * in_filters
    current_tensor = x

    for i in xrange(convolutions):
        with tf.variable_scope("conv_" + str(i + 1)):

            out_ = current_tensor
            if batch_norm:
                out_ = tf.layers.batch_normalization(out_, axis=1, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)

            if activation == "crelu":
                in_filters = 2 * in_filters
                out_ = tf.concat([out_, -out_], axis=1)
            out_ = tf.nn.relu(out_)

            stddev = np.sqrt(2. / in_filters)
            w_pointwise = weight_variable([1, 1, in_filters, out_filters / depthwise_multiplier], stddev, name='weights_pointwise')

            stddev = np.sqrt(2. / (filter_size**2))
            w_depth = weight_variable([filter_size, filter_size, out_filters / depthwise_multiplier, depthwise_multiplier], stddev, name='weights_depthwise')
            
            b = bias_variable([out_filters, 1, 1], name='biases')

            out_ = tf.nn.conv2d(out_, w_pointwise, [1, 1, 1, 1], padding="VALID", data_format=data_format, name="pointwise_conv")
            out_ = tf.nn.depthwise_conv2d_native(out_, w_depth, [1, 1, 1, 1], padding="SAME", data_format=data_format, name="depthwise_conv")
            out_ = out_ + b

            in_filters = out_filters
            current_tensor = out_

    if channel_multiplier != 1:
        x = tf.tile(x, [1, channel_multiplier, 1, 1])
        return x + current_tensor, x

    return x + current_tensor


def upconv_concat_block(x_bottom, x_side, data_format="NHWC"):
    """
    A block which performs the up-convolution of x_bottom and concats it with x_side.

    Parameters
    ----------
    x_bottom: TF tensor
        The input to the upconvolution.
    x_side: TF tensor
        The input to be concatenated.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    upconv_concat_block: TF operation
        The output of the upconv concat block.     
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    #if data_format == "NHWC":
    #    raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    bottom_shape = x_bottom.shape.as_list()
    side_shape = x_side.shape.as_list()

    in_filters = bottom_shape[3]
    out_filters = side_shape[3]
    filter_size = side_shape[1] / bottom_shape[1]

    stddev = np.sqrt(2. / in_filters)
    stride = filter_size

    with tf.name_scope('upconv'):
        w = weight_variable([filter_size, filter_size, out_filters, in_filters], stddev, name='weights')
        b = bias_variable([1,1,out_filters], name='biases')

        y_bottom = upconv2d(x_bottom, w, stride, side_shape, data_format=data_format)
        y_bottom += b
    
    y_total = tf.concat([x_side, y_bottom], 3, name='concat')
    
    return y_total


def upconv_add_block(x_bottom, x_side, data_format="NHWC"):
    """
    A block which performs the up-convolution of x_bottom and adds it to x_side.

    Parameters
    ----------
    x_bottom: TF tensor
        The input to the upconvolution.
    x_side: TF tensor
        The input to be added to the first.
    data_format: string
        The data format, either "NCHW" or "NHWC".

    Returns
    -------
    upconv_add_block: TF operation
        The output of the upconv add block.     
    """

    if data_format not in ["NHWC", "NCHW"]:
        raise ValueError("data_format must be \"NHWC\" or \"NCHW\".")

    #if data_format == "NHWC":
    #    raise NotImplementedError("data_format \"NHWC\" is not yet implemented!")

    bottom_shape = x_bottom.shape.as_list()
    side_shape = x_side.shape.as_list()

    in_filters = bottom_shape[1]
    out_filters = side_shape[1]
    filter_size = side_shape[2] / bottom_shape[2]

    stddev = np.sqrt(2. / in_filters)
    stride = filter_size

    with tf.variable_scope('upconv'):
        w = weight_variable([filter_size, filter_size, out_filters, in_filters], stddev, name='weights')
        b = bias_variable([out_filters, 1, 1], name='biases')
        y_bottom = upconv2d(x_bottom, w, stride, side_shape, data_format=data_format)
        y_bottom += b

    y_total = x_side + y_bottom

    return y_total


# Losses

def weighted_softmax_cross_entropy_loss_with_false_positive_weights(logits, labels, weights, false_positive_factor=0.5):
    """
    Computes the SoftMax Cross Entropy loss with class weights based on the class of each pixel and an additional weight
    for false positive classifications (instances of class 0 classified as class 1).

    Parameters
    ----------
    logits: TF tensor
        The network output before SoftMax.
    labels: TF tensor
        The desired output from the ground truth.
    weights : list of floats
        A list of the weights associated with the different labels in the ground truth.
    false_positive_factor: float
        False positives receive a loss weight of false_positive_factor * label_weights[1], the weight of the class of interest.

    Returns
    -------
    loss : TF float
        The loss.
    weight_map: TF Tensor
        The loss weights assigned to each pixel. Same dimensions as the labels.
    
    """

    with tf.name_scope('loss'):

        logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
        labels = tf.reshape(labels, [-1], name='flatten_labels')

        # get predictions from likelihoods
        prediction = tf.argmax(logits, 1, name='predictions')

        # get maps of class_of_interest pixels
        prediction_map = tf.equal(prediction, 1, name='prediction_map')
        label_map = tf.equal(labels, 1, name='label_map')

        false_positive_map = tf.logical_and(prediction_map, tf.logical_not(label_map), name='false_positive_map')

        label_map = tf.to_float(label_map)
        false_positive_map = tf.to_float(false_positive_map)

        weight_map = label_map * (weights[1] - weights[0]) + weights[0]
        weight_map = tf.add(weight_map, false_positive_map * ((false_positive_factor * weights[1]) - weights[0]), name="combined_weight_map")

        weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

        # compute cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_softmax')

        # apply weights to cross entropy loss
        weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

        # get loss scalar
        loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

    return loss, weight_map


def weighted_softmax_cross_entropy_loss(logits, labels, weights):
    """
    Computes the SoftMax Cross Entropy loss with class weights based on the class of each pixel.

    Parameters
    ----------
    logits: TF tensor
        The network output before SoftMax.
    labels: TF tensor
        The desired output from the ground truth.
    weights : list of floats
        A list of the weights associated with the different labels in the ground truth.

    Returns
    -------
    loss : TF float
        The loss.
    weight_map: TF Tensor
        The loss weights assigned to each pixel. Same dimensions as the labels.
    
    """

    with tf.name_scope('loss'):

        logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
        labels = tf.reshape(labels, [-1], name='flatten_labels')

        weight_map = tf.to_float(tf.equal(labels, 0, name='label_map_0')) * weights[0]
        #for i, weight in enumerate(weights[1:], start=1):
        #    weight_map = weight_map + tf.to_float(tf.equal(labels, i, name='label_map_' + str(i))) * weight
        weight_map = weight_map + tf.to_float(tf.equal(labels, 1, name='label_map_1')) * weights[1]

        weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

        # compute cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_softmax')

        # apply weights to cross entropy loss
        weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

        # add l2 regularization
        train_vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'bias' not in v.name ]) * 0.001

        # add L2 loss to the softmax cross entropy value
        weighted_cross_entropy = tf.add(weighted_cross_entropy, lossL2, name='add_L2loss')

        # get loss scalar
        loss = tf.reduce_mean(weighted_cross_entropy, name='loss')
        

    return loss, weight_map


# Optimizer

def adam_optimizer(loss_op, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=0.1e-08, update_ops=False):
    """
    An adam optimizer which also updates the running averages of any batch normalization layer.

    Parameters
    ----------
    loss_op: TF tensor
        The loss to minimize.
    learning_rate: float
        The learning rate.
    beta1: float
        The beta1 value of the adam optimizer.
    beta2: float
        The beta2 value of the adam optimizer.
    epsilon: float
        The epsilon value of the adam optimizer.
    update_ops: bool
        Whether or not ops from batch normalization need to be updated.

    Returns
    -------
    train_op: TF operation
        The adam optimizer minimizing the loss.
    global_step: TF int
        The current training step.
    """

    with tf.variable_scope('train'):

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)

        if update_ops:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op, global_step=global_step, name='minimize_loss')
        else:
            grads = optimizer.compute_gradients(loss_op)
            train_op = optimizer.apply_gradients(grads, global_step=global_step, name='apply_grads')
            grads_summary_op = tf.summary.merge([tf.summary.histogram("%s_grads" % g[1].name, g[0], collections=['gradients']) for g in grads])
            # train_op = optimizer.minimize(loss_op, global_step=global_step, name='minimize_loss')

    return train_op, global_step, grads_summary_op
