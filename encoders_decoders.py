import tensorflow as tf
import tflearn
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d, global_avg_pool
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers



# ========================================= POINT CLOUD DECODER(should be -ENCODER) ==========================================
'''
Code from @author: optas
Modified for plm
'''
def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d, plm=False):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print 'Building Encoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')

    _range = (n_layers - 1) if plm else n_layers
    for i in xrange(_range):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print layer

    if closing is not None:
        layer = closing(layer)
        print layer

    if not plm:
        return layer
    else:
        name = 'encoder_z_mean'
        scope = expand_scope_by_name(scope, name)
        z_mean = fully_connected(layer, n_filters[-1], activation='linear', weights_init='xavier', name=name, regularizer=None, weight_decay=weight_decay, reuse=reuse, scope=scope)
        name = 'encoder_z_log_sigma_sq'
        scope = expand_scope_by_name(scope, name)
        z_log_sigma_sq = fully_connected(layer, n_filters[-1], activation='softplus', weights_init='xavier', name=name, regularizer=None, weight_decay=weight_decay, reuse=reuse, scope=scope)
        return z_mean, z_log_sigma_sq

# ========================================= POINT CLOUD DECODER ==========================================
'''
Code from @author: optas
'''
def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu6,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

# ========================================= POINT CLOUD DECODER ==========================================
'''
Code from @author: optas
'''
def decoder_with_fc_only_v1(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print 'Building Decoder'

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in xrange(0, n_layers - 1):
        name = 'decoder_fc_v1' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if b_norm:
            name += '_bnorm_v1'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_v1' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

    if b_norm_finish:
        name += '_bnorm_v1'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

    if verbose:
        print layer
        print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

# ========================================= POINT CLOUD DECODER conv==========================================

def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print 'Building Decoder'

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in xrange(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list())

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print layer
            print 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n'

    return layer

# ========================================= IMAGE ENCODER ==========================================
def image_encoder_div(img_inp, FLAGS):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''

    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    #x = squeeze_excitation_layer(x,64,4,'se_1')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x = squeeze_excitation_layer(x,128,4,'se_2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x = squeeze_excitation_layer(x,256,4,'se_3')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    z_mean = tflearn.layers.core.fully_connected(x, FLAGS.bottleneck, activation='linear', weight_decay=1e-3,regularizer='L2')
    z_log_sigma_sq = tflearn.layers.core.fully_connected(x, FLAGS.bottleneck, activation='linear', weight_decay=1e-3,regularizer='L2')
    return z_mean, z_log_sigma_sq

# ========================================= IMAGE ENCODER ==========================================
def image_encoder(img_inp, FLAGS):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''
    #r = np.random.normal(0,1,128*24)
    #r = r.astype(np.float32)
    #r = tf.reshape(r,shape=[-1,128])
    #r = tflearn.layers.core.fully_connected(r, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    #r = tflearn.layers.core.fully_connected(r, 1024*3, activation='relu', weight_decay=1e-3, regularizer='L2')
    # 32*32*3 = 1024*3
    #r = tf.reshape(r,shape=[-1,32,32,3])
    #r = tflearn.layers.conv.conv_2d(r, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,    regularizer='L2')  
    # [B*5, 24, 32, 32]
    #r = tflearn.layers.conv.conv_2d(r, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')


    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    #x = tf.concat([x, r], axis=3)
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')


    x_latent=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='linear',weight_decay=1e-3,regularizer='L2')
    return x_latent

# ========================================= IMAGE ENCODER  with r ======================================
def image_encoder_r(img_inp, FLAGS):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''
    r = np.random.normal(0,1,128*FLAGS.batch_size)
    r = r.astype(np.float32)
    r = tf.reshape(r,shape=[-1,128])
    r = tflearn.layers.core.fully_connected(r, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    r = tflearn.layers.core.fully_connected(r, 1024*3, activation='relu', weight_decay=1e-3, regularizer='L2')
    # 32*32*3 = 1024*3
    r = tf.reshape(r,shape=[-1,32,32,3])
    r = tflearn.layers.conv.conv_2d(r, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,    regularizer='L2')  
    # [B*5, 24, 32, 32]
    r = tflearn.layers.conv.conv_2d(r, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')


    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x = tf.concat([x, r], axis=3)
    #x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

    if FLAGS.mode == 'lm':
        x_latent=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='linear',weight_decay=1e-3,regularizer='L2')
        return x_latent
    elif FLAGS.mode == 'plm':
        z_mean = tflearn.layers.core.fully_connected(x, FLAGS.bottleneck, activation='linear', weight_decay=1e-3,regularizer='L2')
        z_log_sigma_sq = tflearn.layers.core.fully_connected(x, FLAGS.bottleneck, activation='linear', weight_decay=1e-3,regularizer='L2')
        return z_mean, z_log_sigma_sq

# ========================================= IMAGE ENCODER ==========================================
def image_encoder_se_pure(img_inp, FLAGS):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''
    #r = np.random.normal(0,1,128*24)
    #r = r.astype(np.float32)
    #r = tf.reshape(r,shape=[-1,128])
    #r = tflearn.layers.core.fully_connected(r, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    #r = tflearn.layers.core.fully_connected(r, 1024*3, activation='relu', weight_decay=1e-3, regularizer='L2')
    # 32*32*3 = 1024*3
    #r = tf.reshape(r,shape=[-1,32,32,3])
    #r = tflearn.layers.conv.conv_2d(r, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,    regularizer='L2')  
    # [B*5, 24, 32, 32]
    #r = tflearn.layers.conv.conv_2d(r, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')


    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    #x = squeeze_excitation_layer(x,64,4,'se_1')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x = squeeze_excitation_layer(x,128,4,'se_2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x = squeeze_excitation_layer(x,256,4,'se_3')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')


    x_latent=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='linear',weight_decay=1e-3,regularizer='L2')
    return x_latent


# ========================================= IMAGE ENCODER ==========================================
def image_encoder_se(img_inp, FLAGS, training_flag):
    '''
    Input:
        img_inp: tf placeholder of shape (B, HEIGHT, WIDTH, 3) corresponding to RGB image
    Returns:
        x_latent: tensor of shape (B, FLAGS.bottleneck) corresponding to the predicted latent vector
    Description:
        Main Architecture for Latent Matching Network
    '''
    #r = np.random.normal(0,1,128*24)
    #r = r.astype(np.float32)
    #r = tf.reshape(r,shape=[-1,128])
    #r = tflearn.layers.core.fully_connected(r, 256, activation='relu', weight_decay=1e-3, regularizer='L2')
    #r = tflearn.layers.core.fully_connected(r, 1024*3, activation='relu', weight_decay=1e-3, regularizer='L2')
    # 32*32*3 = 1024*3
    #r = tf.reshape(r,shape=[-1,32,32,3])
    #r = tflearn.layers.conv.conv_2d(r, 32, (3, 3), strides=1, activation='relu', weight_decay=1e-5,    regularizer='L2')  
    # [B*5, 24, 32, 32]
    #r = tflearn.layers.conv.conv_2d(r, 128, (3, 3), strides=1, activation='relu', weight_decay=1e-5, regularizer='L2')


    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    input_x=x
    x = split_layer(x,training_flag,depth=128, stride=1, layer_name='split_layer__1')
    x = transition_layer(x, training_flag, out_dim=128, scope='trans_layer__1')
    x = squeeze_excitation_layer(x, out_dim=128, ratio=4, layer_name='squeeze_layer__1')
    x = Relu(x + input_x)
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #8 8
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')


    x_latent=tflearn.layers.core.fully_connected(x,FLAGS.bottleneck,activation='linear',weight_decay=1e-3,regularizer='L2')
    return x_latent


#the ratio prefer set to 4 ---------------------------------------
def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale


def Global_Average_Pooling(x):
    return global_avg_pool(x, name="SE_Global_average_pooling")

def Fully_connected(x, units, layer_name='SE_fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)


#-------------------------------------

def split_layer(input_x, is_training,depth, stride, layer_name):
    with tf.name_scope(layer_name) :
        layers_split = list()
        for i in range(3) :
            splits = transform_layer(input_x, is_training, depth=depth, stride=stride, scope=layer_name + '_splitN_' + str(i))
            layers_split.append(splits)

        return Concatenation(layers_split)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def transform_layer(x, is_training,depth, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, is_training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, is_training, scope=scope+'_batch2')
            x = Relu(x)
            return x

def transition_layer(x, is_training, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, is_training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

def Batch_Normalization(x,is_training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(is_training,
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=True))

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network