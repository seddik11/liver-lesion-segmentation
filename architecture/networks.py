import numpy as np
import tensorflow as tf

import layers


def parameter_efficient(in_channels=1, out_channels=2, start_filters=64, input_side_length=256, depth=4, res_blocks=2, filter_size=3, sparse_labels=True, batch_size=1, activation="cReLU", batch_norm=True):
    """
    Creates the graph for the parameter efficient variant of the U-Net and sets up the appropriate input and output placeholder.

    Parameters
    ----------
    in_channels: int
        The depth of the input.
    out_channels: int
        The depth of number of classes of the output.
    start_filters : int
        The number of filters in the first convolution.
    input_side_length: int
        The side length of the square input.
    depth: int
        The depth of the U-part of the network. This is equal to the number of max-pooling layers.
    res_blocks: int
        The number of residual blocks in between max-pooling layers on the down-path and in between up-convolutions on the up-path.
    filter_size: int
        The width and height of the filter. The receptive field.
    sparse_labels: bool
        If true, the labels are integers, one integer per pixel, denoting the class that that pixel belongs to. If false, labels are one-hot encoded.
    batch_size: int
        The training batch size.
    activation: string
        Either "ReLU" for the standard ReLU activation or "cReLU" for the concatenated ReLU activation function.
    batch_norm: bool
        Whether to use batch normalization or not.

    Returns
    -------
    inputs : TF tensor
        The network input.
    logits: TF tensor
        The network output before SoftMax.
    ground_truth: TF tensor
        The desired output from the ground truth.
    keep_prob: TF float
        The TF variable holding the keep probability for drop out layers.
    training_bool: TF bool
        The TF variable holding the boolean value, which switches batch normalization to training or inference mode.    
    """

    activation = str.lower(activation)
    if activation not in ["relu", "crelu"]:
        raise ValueError("activation must be \"ReLU\" or \"cReLU\".")

    pool_size = 2

    # Define inputs and helper functions #

    with tf.variable_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, input_side_length, input_side_length, in_channels), name='inputs')
        if sparse_labels:
            ground_truth = tf.placeholder(tf.int32, shape=(batch_size, input_side_length, input_side_length), name='labels')
        else:
            ground_truth = tf.placeholder(tf.float32, shape=(batch_size, input_side_length, input_side_length, out_channels), name='labels')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        training = tf.placeholder(tf.bool, shape=[], name="training")

        network_input = tf.transpose(inputs, perm=[0, 3, 1, 2])

    # [conv -> conv -> max pool -> drop out] + parameter updates
    def step_down(name, input_, filter_size=3, res_blocks=2, keep_prob=1., training=False):

        with tf.variable_scope(name):
            
            with tf.variable_scope("res_block_0"):
                conv_out, tiled_input = layers.res_block(input_, filter_size, channel_multiplier=2, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            for i in xrange(1, res_blocks):
                with tf.variable_scope("res_block_" + str(i)):
                    conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            conv_out = conv_out + tiled_input

            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            
            bottom_out = layers.dropout(pool_out, keep_prob)
            side_out = layers.dropout(conv_out, keep_prob)

        return bottom_out, side_out

    # parameter updates + [upconv and concat -> drop out -> conv -> conv]
    def step_up(name, bottom_input, side_input, filter_size=3, res_blocks=2, keep_prob=1., training=False):

        with tf.variable_scope(name):
            added_input = layers.upconv_add_block(bottom_input, side_input, data_format="NCHW")

            conv_out = added_input
            for i in xrange(res_blocks):
                with tf.variable_scope("res_block_" + str(i)):
                    conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
            
            result = layers.dropout(conv_out, keep_prob)

        return result

    # Build the network #

    with tf.variable_scope('contracting'):

        outputs = []

        with tf.variable_scope("step_0"):

            # Conv 1
            in_filters = in_channels
            out_filters = start_filters

            stddev = np.sqrt(2. / (filter_size**2 * in_filters))
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_filters], stddev=stddev, name="weights")

            out_ = tf.nn.conv2d(network_input, w, [1, 1, 1, 1], padding="SAME", data_format="NCHW")
            out_ = out_ + layers.bias_variable([out_filters, 1, 1], name='biases')

            # Batch Norm 1
            if batch_norm:
                out_ = tf.layers.batch_normalization(out_, axis=1, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)

            in_filters = out_filters

            # concatenated ReLU
            if activation == "crelu":
                out_ = tf.concat([out_, -out_], axis=1)
                in_filters = 2 * in_filters
            out_ = tf.nn.relu(out_)

            # Conv 2
            stddev = np.sqrt(2. / (filter_size**2 * in_filters))
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_filters], stddev=stddev, name="weights")

            out_ = tf.nn.conv2d(out_, w, [1, 1, 1, 1], padding="SAME", data_format="NCHW")
            out_ = out_ + layers.bias_variable([out_filters, 1, 1], name='biases')

            # Res Block 1
            conv_out = layers.res_block(out_, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")

            pool_out = layers.max_pool(conv_out, pool_size, data_format="NCHW")
            
            bottom_out = layers.dropout(pool_out, keep_prob)
            side_out = layers.dropout(conv_out, keep_prob)

            outputs.append(side_out)

        # Build contracting path
        for i in xrange(1, depth):
            bottom_out, side_out = step_down('step_' + str(i), bottom_out, filter_size=filter_size, res_blocks=res_blocks, keep_prob=keep_prob, training=training)
            outputs.append(side_out)

    # Bottom [conv -> conv]
    with tf.variable_scope('step_' + str(depth)):

        with tf.variable_scope("res_block_0"):
            conv_out, tiled_input = layers.res_block(bottom_out, filter_size, channel_multiplier=2, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
        for i in xrange(1, res_blocks):
            with tf.variable_scope("res_block_" + str(i)):
                conv_out = layers.res_block(conv_out, filter_size, channel_multiplier=1, depthwise_multiplier=2, convolutions=2, training=training, activation=activation, batch_norm=batch_norm, data_format="NCHW")
        
        conv_out = conv_out + tiled_input
        current_tensor = layers.dropout(conv_out, keep_prob)

    with tf.variable_scope('expanding'):

        # Set initial parameter
        outputs.reverse()

        # Build expanding path
        for i in xrange(depth):
            current_tensor = step_up('step_' + str(depth + i + 1), current_tensor, outputs[i], filter_size=filter_size, res_blocks=res_blocks, keep_prob=keep_prob, training=training)
 
    # Last layer is a 1x1 convolution to get the predictions
    # We don't want an activation function for this one (softmax will be applied later), so we're doing it manually
    in_filters = current_tensor.shape.as_list()[1]
    stddev = np.sqrt(2. / in_filters)

    with tf.variable_scope('classification'):

        w = layers.weight_variable([1, 1, in_filters, out_channels], stddev, name='weights')
        b = layers.bias_variable([out_channels, 1, 1], name='biases')

        conv = tf.nn.conv2d(current_tensor, w, strides=[1, 1, 1, 1], padding="SAME", data_format="NCHW", name='conv')
        logits = conv + b

        logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    return inputs, logits, ground_truth, keep_prob, training

def Fully_Dilted_Convolutions_For_Liver_Segmentation(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, batch_size=1):
    
    """
    our Model
    """
    # parameters

    # Define inputs and helper functions #
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            layer = inputs
            out_filters = start_filters
            dilation_rate = 1
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=1, padding="SAME")

        with tf.name_scope('increasing_rates'):
            for i in xrange(1,depth + 1):
                dilation_rate *= dilation_factor
                out_filters += growth_rate
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=i, padding="SAME")
        
        with tf.name_scope('decreasing_rates'):
            for i in xrange(depth + 1,depth * 2 + 1):
                dilation_rate /= dilation_factor
                out_filters -= growth_rate
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=i, padding="SAME")
        
        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training

# def Fully_Dense_Dilted_Convolutions_For_Liver_Segmentation(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor = 2, growth_rate=12, side_length=512, convolutions=1, filter_size=3, batch_size=1):

#     with tf.variable_scope('inputs'):
#         inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
#         ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
#         training = tf.placeholder(tf.bool, shape=[], name="training")
#         keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

#     with tf.variable_scope('network_architecture'):
#         layer = inputs
#         dilation_rate = dilation_factor
#         for index in xrange(depth):
#             layer = layers.atrous_with_dense(layer, filter_size=filter_size, growth_rate=growth_rate, dilation_rate=dilation_rate, padding="SAME", training=training, index=index)
#             dilation_rate *= dilation_factor
    
#     with tf.variable_scope('classification'):
#         in_filters = layer.shape.as_list()[3]
#         layer = tf.layers.batch_normalization(layer, axis=3, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm", fused=True)
#         layer = tf.nn.relu(layer, name='relu')
#         w = layers.weight_fixed([filter_size, filter_size, in_filters, out_channels],name='weights')
#         b = layers.bias_variable([1, 1, out_channels], name='biases')
#         conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
#         logits = conv + b

#     return inputs, logits, ground_truth, keep_prob, training 

def Fully_Dense_Dilted_Convolutions_For_Liver_Segmentation(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, batch_size=1):
    
    """
    our Model
    """
    # parameters

    # Define inputs and helper functions #
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            layer = inputs
            out_filters = start_filters
            dilation_rate = 1
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=0, padding="SAME", batch_norm=False, training=training, dense=False)

        outputs = []
        concats = []
        outputs.append(layer)

        with tf.name_scope('increasing_rates'):
            for i in xrange(1, depth+1):
                # dilation_rate *= dilation_factor
                # out_filters += growth_rate
                n = 1
                sublayers = []
                for j in reversed(xrange(i)):
                    layer = layers.atrous(layer, filter_size=filter_size, out_filters=growth_rate, dilation_rate=n, index=i*10+j, padding="SAME", batch_norm=False, training=training, dense=False, outputs=outputs, merge=j)
                    if n == 1:
                        concats.append(layer)
                    n *= dilation_factor
                    sublayers.append(layer)
                layer = tf.concat(sublayers, 3)
                outputs.append(layer)

        # with tf.name_scope('bridge'):
        #     dilation_rate *= dilation_factor
        #     out_filters += growth_rate
        #     layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=depth, padding="SAME", batch_norm=False, training=training, dense=False)

            # layer = layers.dropout(layer, 1.)

        concats.reverse()
        outputs = []
        outputs.append(layer)
        
        with tf.name_scope('decreasing_rates'):
            j = 0
            for i in xrange(depth+1, depth*2):
                # dilation_rate /= dilation_factor
                # out_filters -= growth_rate
                sublayers = []
                n = 1
                sublayers.append(concats[j])
                for k in reversed(xrange(j+1)):
                    layer = layers.atrous(layer, filter_size=filter_size, out_filters=growth_rate, dilation_rate=n, index=i*10+k, padding="SAME", batch_norm=False, training=training, dense=False, outputs=outputs, merge=k)
                    sublayers.append(layer)
                    n *= dilation_factor
                j += 1
                layer = tf.concat(sublayers, 3)
                outputs.append(layer)

        with tf.name_scope('exit'):
            # dilation_rate /= dilation_factor
            # out_filters -= growth_rate
            layer = outputs[-1]
            layer = tf.concat([concats[-1], layer], 3)
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=start_filters, dilation_rate=1, index=depth*2*10, padding="SAME", batch_norm=False, training=training)

            # layer = layers.dropout(layer, 1.)

        
        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            # layer = tf.layers.batch_normalization(layer, axis=3, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm"+"soft", fused=True)
            # layer = tf.nn.relu(layer, name='relu'+"loss")
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training


def Modified_Fully_Dense_Dilted_Convolutions_For_Liver_Segmentation(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, batch_size=1):
    
    """
    our Model
    """
    # parameters

    # Define inputs and helper functions #
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            layer = inputs
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=24, dilation_rate=1, index=0, padding="SAME", batch_norm=False, training=training, dense=True)

        with tf.name_scope('increasing_rates'):
            with tf.name_scope('lesions_block'):
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=48, dilation_rate=2, index=1, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=72, dilation_rate=4, index=2, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=96, dilation_rate=8, index=3, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=120, dilation_rate=16, index=4, padding="SAME", batch_norm=False, training=training, dense=True)

            # with tf.name_scope('spatial_block'):
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=144, dilation_rate=32, index=5, padding="SAME", batch_norm=False, training=training, dense=True)
                # layer = layers.atrous(layer, filter_size=filter_size, out_filters=12, dilation_rate=64, index=6, padding="SAME", batch_norm=False, training=training, dense=True)

        with tf.name_scope('bridge'):
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=168, dilation_rate=1, index=7, padding="SAME", batch_norm=False, training=training, dense=False)
            layer = tf.concat([inputs, layer], 3)

            # layer = layers.dropout(layer, 1.)

        with tf.name_scope('decreasing_rates'):
            with tf.name_scope('spatial_block'):
                # layer = layers.atrous(layer, filter_size=filter_size, out_filters=12, dilation_rate=64, index=8, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=144, dilation_rate=32, index=9, padding="SAME", batch_norm=False, training=training, dense=True)

            with tf.name_scope('lesions_block'):
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=120, dilation_rate=16, index=10, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=96, dilation_rate=8, index=11, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=72, dilation_rate=4, index=12, padding="SAME", batch_norm=False, training=training, dense=True)
                layer = layers.atrous(layer, filter_size=filter_size, out_filters=48, dilation_rate=2, index=13, padding="SAME", batch_norm=False, training=training, dense=True)


        with tf.name_scope('exit'):
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=24, dilation_rate=1, index=14, padding="SAME", batch_norm=False, training=training)

            # layer = layers.dropout(layer, 1.)

        
        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            # layer = tf.layers.batch_normalization(layer, axis=3, momentum=0.999, center=True, scale=True, training=training, trainable=True, name="batch_norm"+"soft", fused=True)
            # layer = tf.nn.relu(layer, name='relu'+"loss")
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training

def Fully_Dense_Dilted_Convolutions_Increasing_Module(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, batch_size=1):
    
    """
    our Model
    """
    # parameters

    # Define inputs
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            layer = inputs
            out_filters = start_filters
            dilation_rate = 1
            layer = layers.atrous_with_dense(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=0, padding="SAME")

        with tf.name_scope('increasing_rates'):
            for i in xrange(1, depth):
                dilation_rate *= dilation_factor
                out_filters += growth_rate
                layer = layers.atrous_with_dense(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=i, padding="SAME")

        with tf.name_scope('bridge'):
            dilation_rate *= dilation_factor
            out_filters += growth_rate 
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=depth, padding="SAME")

         # with tf.name_scope('exit'):
         #     for i in xrange(depth + 1, depth + 3)
         #     dilation_rate = 1
         #     out_filters -= growth_rate
         #     layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=depth*2, padding="SAME")
        
        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training


def Fully_Dense_Dilted_Convolutions_Decreasing_Module(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, batch_size=1):
    
    """
    our Model
    """
    # parameters

    # Define inputs
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            layer = inputs
            out_filters = start_filters
            dilation_rate = 1
            layer = layers.atrous_with_dense(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=0, padding="SAME")
            dilation_rate = 64

        with tf.name_scope('decreasing_rates'):
            for i in xrange(1,depth + 1):
                dilation_rate /= dilation_factor
                out_filters += growth_rate
                layer = layers.atrous_with_dense(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, index=i, padding="SAME")

        with tf.name_scope('exit'):
            dilation_rate /= dilation_factor
            out_filters += growth_rate
            layer = layers.atrous(layer, filter_size=filter_size, out_filters=out_filters, dilation_rate=1, index=depth*2, padding="SAME")
        
        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training

def losange(in_channels=5, out_channels=2, start_filters=12, depth=5, dilation_factor=2, growth_rate=12, side_length=512, convolutions=2, filter_size=3, pool_size=2, batch_size=1):
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        training = tf.placeholder(tf.bool, shape=[], name="training")
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
    
    concats = []

    with tf.name_scope('network_architecture'):
        with tf.name_scope('gate'):
            dilation_rate = 1
            out_filters = growth_rate
            layer = (layers.atrous(inputs, filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, padding="SAME"),
                    layers.conv(inputs, out_filters=24, filter_size=filter_size, stride=1, padding="SAME", name="conv"))
            concats.append(layer)

        x_layer = layer[0]
        with tf.name_scope('extracting'):
            for i in xrange(1, depth):
                with tf.name_scope('layer_' + str(i)):
                    dilation_rate *= dilation_factor
                    out_filters += growth_rate
                    layer = layers.spread_down((x_layer, layer[1]), filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, pool_size=pool_size, dense=False)
                    layer = layers.ffru_down(layer, pool_size=pool_size, index=i)
                    x_layer = tf.concat([x_layer, layer[0]], 3)
                    concats.append(layer)

        with tf.name_scope('bridge'):
            dilation_rate *= dilation_factor
            out_filters += growth_rate
            layer = layers.spread_down((x_layer, layer[1]), filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, pool_size=pool_size, dense=False)
            layer = layers.ffru_down(layer, pool_size=pool_size, index=depth)
            concats.reverse()

        x_layer = layer[0]
        with tf.name_scope('projecting'):
            j = 0
            for i in xrange(depth+1, depth*2):
                with tf.name_scope('layer_' + str(i)):
                    dilation_rate /= dilation_factor
                    out_filters -= growth_rate
                    layer = layers.spread_up((x_layer, layer[1]), filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, dense=False, concats=concats[j])
                    layer = layers.ffru_up(layer, filter_size=filter_size)
                    x_layer = tf.concat([x_layer, layer[0]], 3)
                    j += 1

        with tf.name_scope('exit'):
            dilation_rate /= dilation_factor
            out_filters -= growth_rate
            layer = layers.spread_up((x_layer, layer[1]), filter_size=filter_size, out_filters=out_filters, dilation_rate=dilation_rate, dense=False, concats=concats[j])
            layer = layers.ffru_up(layer, filter_size=filter_size)
            layer = tf.concat(layer, 3)
            layer = layers.conv(layer, out_filters=out_filters, filter_size=3, stride=1, padding="SAME", name="conv")

        in_filters = layer.shape.as_list()[3]
        stddev = np.sqrt(2. / in_filters)
        with tf.name_scope('classification'):
            w = layers.weight_variable([filter_size, filter_size, in_filters, out_channels], stddev, name='weights')
            b = layers.bias_variable([1, 1, out_channels], name='biases')
            conv = tf.nn.atrous_conv2d(layer, w, 1, padding="SAME", name='atrous_conv2d')
            logits = conv + b

    return inputs, logits, ground_truth, keep_prob, training

    

def unet(in_channels=1, out_channels=2, start_filters=64, side_length=512, depth=4, convolutions=2, filter_size=3, sparse_labels=True, batch_size=1):
    """
    Creates the graph for the standard U-Net and sets up the appropriate input and output placeholder.

    Parameters
    ----------
    in_channels: int
        The depth of the input.
    out_channels: int
        The depth of number of classes of the output.
    start_filters : int
        The number of filters in the first convolution.
    side_length: int
        The side length of the square input.
    depth: int
        The depth of the U-part of the network. This is equal to the number of max-pooling layers.
    convolutions: int
        The number of convolutions in between max-pooling layers on the down-path and in between up-convolutions on the up-path.
    filter_size: int
        The width and height of the filter. The receptive field.
    sparse_labels: bool
        If true, the labels are integers, one integer per pixel, denoting the class that that pixel belongs to. If false, labels are one-hot encoded.
    batch_size: int
        The training batch size.

    Returns
    -------
    inputs : TF tensor
        The network input.
    logits: TF tensor
        The network output before SoftMax.
    ground_truth: TF tensor
        The desired output from the ground truth.
    keep_prob: TF float
        The TF variable holding the keep probability for drop out layers.  
    """

    pool_size = 2
    padding = "SAME"

    # Define inputs and helper functions #
    with tf.variable_scope('inputs'):
        inputs = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, in_channels), name='inputs')
        if sparse_labels:
            ground_truth = tf.placeholder(tf.int32, shape=(batch_size, side_length, side_length), name='labels')
        else:
            ground_truth = tf.placeholder(tf.float32, shape=(batch_size, side_length, side_length, out_channels), name='labels')
        keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        training = tf.placeholder(tf.bool, shape=[], name="training")

        #network_input = tf.transpose(inputs, perm=[0, 3, 1, 2])

        network_input = inputs

    # [conv -> conv -> max pool -> drop out] + parameter updates
    def step_down(name, _input):

        with tf.variable_scope(name):
            conv_out = layers.conv_block(_input, filter_size, channel_multiplier=2, convolutions=convolutions, padding=padding)
            pool_out = layers.max_pool(conv_out, pool_size)
            result = layers.dropout(pool_out, keep_prob)

        return result, conv_out

    # parameter updates + [upconv and concat -> drop out -> conv -> conv]
    def step_up(name, bottom_input, side_input):

        with tf.variable_scope(name):
            concat_out = layers.upconv_concat_block(bottom_input, side_input)
            drop_out = layers.dropout(concat_out, keep_prob)
            result = layers.conv_block(drop_out, filter_size, channel_multiplier=0.5, convolutions=convolutions, padding=padding)

        

        return result

    # Build the network #

    with tf.variable_scope('contracting'):

        # Set initial parameters
        outputs = []

        # Build contracting path
        with tf.variable_scope("step_0"):
            conv_out = layers.conv_block(network_input, filter_size, out_filters=start_filters, convolutions=convolutions, padding=padding)
            pool_out = layers.max_pool(conv_out, pool_size)
            current_tensor = layers.dropout(pool_out, keep_prob)
            outputs.append(conv_out)

        for i in xrange(1, depth):
            current_tensor, conv_out = step_down("step_" + str(i), current_tensor)
            outputs.append(conv_out)

    # Bottom [conv -> conv]
    with tf.variable_scope("step_" + str(depth)):
        current_tensor = layers.conv_block(current_tensor, filter_size, channel_multiplier=2, convolutions=convolutions, padding=padding)

    with tf.variable_scope("expanding"):

        # Set initial parameter
        outputs.reverse()

        # Build expanding path
        for i in xrange(depth):
            current_tensor = step_up("step_" + str(depth + i + 1), current_tensor, outputs[i])

    # Last layer is a 1x1 convolution to get the predictions
    # We don't want an activation function for this one (softmax will be applied later), so we're doing it manually
    in_filters = current_tensor.shape.as_list()[3]
    stddev = np.sqrt(2. / in_filters)

    with tf.variable_scope("classification"):

        weight = layers.weight_variable([1, 1, in_filters, out_channels], stddev, name="weights")
        bias = layers.bias_variable([1,1,out_channels], name="biases")

        conv = tf.nn.conv2d(current_tensor, weight, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        logits = conv + bias

        #logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    return inputs, logits, ground_truth, keep_prob, training


def get_output_side_length(side_length, depth, convolutions, filter_size, pool_size):
    """
    Computes the output side length for a standard U-Net without padded convolutions.

    Parameters
    ----------
    side_length: int
        The side length of the square input.
    depth: int
        The depth of the U-part of the network. This is equal to the number of max-pooling layers.
    convolutions: int
        The number of convolutions in between max-pooling layers on the down-path and in between up-convolutions on the up-path.
    filter_size: int
        The width and height of the filter. The receptive field.
    pool_size: int
        The width and height of the filter. The receptive field.
    batch_size: int
        The training batch size.
    padded_convolutions: bool
        Whether to pad the input to keep the side length constant through convolutional layers or not.
        If no padding is used, the side length decreases with every convolution.

    Returns
    -------
    inputs : TF tensor
        The network input.
    logits: TF tensor
        The network output before SoftMax.
    ground_truth: TF tensor
        The desired output from the ground truth.
    keep_prob: TF float
        The TF variable holding the keep probability for drop out layers.  
    """

    for i in xrange(depth - 1):

        for j in xrange(convolutions):
            side_length -= (filter_size - 1)
            if side_length < 0:
                raise ValueError("Input side length too small. Side length < 0 in contracting path after {} max pooling layers plus {} convolution.".format(i, j + 1))

        if (side_length % pool_size) != 0:
            raise ValueError("problem with input side length. Side length not divisible by pool size {}. Side length is {} before max pooling layer {}.".format(pool_size, side_length, i + 1))
        else:
            side_length /= pool_size

    for j in xrange(convolutions):
        side_length -= (filter_size - 1)
        if side_length < 0:
            raise ValueError("Input side length too small. Side length < 0 at bottom layer after {} convolution.".format(j + 1))

    for i in xrange(depth - 1):
        side_length *= pool_size
        side_length -= convolutions * (filter_size - 1)

    return side_length
