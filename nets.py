from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 10
DISP_SCALING_RESNET50 = 5
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                        normalizer_fn=None, activation_fn=None)
                    
                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points

def disp_net(tgt_image, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points

# def dispRes_net(tgt_image, is_training=True):
#     H = tgt_image.get_shape()[1].value
#     W = tgt_image.get_shape()[2].value
#     batch_norm_params = {'is_training': is_training}
#     with tf.variable_scope('depth_net') as sc:
#         end_points_collection = sc.original_name_scope + '_end_points'
#         with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                             normalizer_fn=slim.batch_norm,
#                             normalizer_params = batch_norm_params,
#                             weights_regularizer=slim.l2_regularizer(0.0001),
#                             activation_fn=tf.nn.relu,
#                             outputs_collections=end_points_collection):
#             cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
#             # cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
#             pool1 = maxpool(cnv1,3)
#             cnv2 = resblock(pool1,64,3)             #H/8 - 256D
#             cnv3 = resblock(cnv2,128,3)             #H/16 - 512D
#             cnv4 = resblock(cnv3,256,6)             #H/32 - 1024D
#             cnv5 = resblock(cnv4,512,3)             #H/64 - 2048D

#             skip1 = cnv1
#             skip2 = pool1
#             skip3 = cnv2
#             skip4 = cnv3
#             skip5 = cnv4

#             upcnv6 = slim.conv2d_transpose(cnv5, 512, [3, 3], stride=2, scope='upcnv6')
#             upcnv6 = resize_like(upcnv6, skip5)
#             i6_in  = tf.concat([upcnv6, skip5], axis=3)
#             icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

#             upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
#             upcnv5 = resize_like(upcnv5, skip4)
#             i5_in  = tf.concat([upcnv5, skip4], axis=3)
#             icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

#             upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
#             upcnv4 = resize_like(upcnv4, skip3)
#             i4_in  = tf.concat([upcnv4, skip3], axis=3)
#             icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
#             disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1,
#                 activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
#             disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

#             upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
#             # upcnv3 = resize_like(upcnv3, skip2)
#             i3_in  = tf.concat([upcnv3, skip2, disp4_up], axis=3)
#             icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
#             disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1,
#                 activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
#             disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

#             upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
#             i2_in  = tf.concat([upcnv2, skip1, disp3_up], axis=3)
#             icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
#             disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1,
#                 activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
#             disp2_up = tf.image.resize_bilinear(disp2, [H, W])

#             upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
#             i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
#             icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
#             disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1,
#                 activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

#             end_points = utils.convert_collection_to_dict(end_points_collection)
#             return [disp1, disp2, disp3, disp4], end_points

# def maxpool(x, kernel_size):
#     p = np.floor((kernel_size - 1) / 2).astype(np.int32)
#     p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
#     return slim.max_pool2d(p_x, kernel_size,scope='pool1')

# def resconv(x, num_layers, stride):
#     # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
#     # but we preserve it here for consistency with Godard's implementation.
#     do_proj = tf.shape(x)[3] != num_layers or stride == 2
#     shortcut = []
#     conv1 = conv(x,         num_layers, 1, 1)
#     conv2 = conv(conv1,     num_layers, 3, stride)
#     conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
#     if do_proj:
#         shortcut = conv(x, 4 * num_layers, 1, stride, None)
#     else:
#         shortcut = x
#     return tf.nn.relu(conv3 + shortcut)

# def resblock(x, num_layers, num_blocks):
#     out = x
#     for i in range(num_blocks - 1):
#         out = resconv(out, num_layers, 1)
#     out = resconv(out, num_layers, 2)
#     return out
# def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
#     p = np.floor((kernel_size - 1) / 2).astype(np.int32)
#     p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
#     return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

# def maxpool(x, kernel_size):
#     p = np.floor((kernel_size - 1) / 2).astype(np.int32)
#     p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
#     return slim.max_pool2d(p_x, kernel_size)

# def resnet_v1_50(inputs,
#                  num_classes=None,
#                  is_training=True,
#                  global_pool=True,
#                  output_stride=None,
#                  spatial_squeeze=True,
#                  store_non_strided_activations=False,
#                  min_base_depth=8,
#                  depth_multiplier=1,
#                  reuse=None,
#                  scope='resnet_v1_50'):
#   """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
#   depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
#   blocks = [
#       resnet_v1_block('block1', base_depth=depth_func(64), num_units=3,
#                       stride=2),
#       resnet_v1_block('block2', base_depth=depth_func(128), num_units=4,
#                       stride=2),
#       resnet_v1_block('block3', base_depth=depth_func(256), num_units=6,
#                       stride=2),
#       resnet_v1_block('block4', base_depth=depth_func(512), num_units=3,
#                       stride=1),
#   ]
#   return resnet_v1(inputs, blocks, num_classes, is_training,
#                    global_pool=global_pool, output_stride=output_stride,
#                    include_root_block=True, spatial_squeeze=spatial_squeeze,
#                    store_non_strided_activations=store_non_strided_activations,
#                    reuse=reuse, scope=scope)

def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv

def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

def dispRes_net(inputs, is_training, var_scope='depth_resnet'):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = conv(inputs, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # DECODING
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_disp_resnet50(iconv4)
            upred4  = upsample_nn(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_disp_resnet50(iconv3)
            upred3  = upsample_nn(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_disp_resnet50(iconv2)
            upred2  = upsample_nn(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_disp_resnet50(iconv1)

            return [pred1, pred2, pred3, pred4]
