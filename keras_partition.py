import sys
sys.path.append(".")
import tensorflow as tf
#import tensorflow.keras.applications.inception_v3 as inception_v3
import model_zoo.keras.inception_v3_change_layer as inception_v3
#import tensorflow.keras.applications.resnet50 as resnet50
#import model_zoo.keras.inception_v3_bk as inception_v3
import vis.utils as utils
import numpy as np
import tensorflow.keras.layers as layers
class InceptionWJ(tf.keras.Model):
    def conv2d_bn(self,x,filters,num_row,num_col,padding='same',strides=(1, 1),name=None):
        """Utility function to apply conv + BN.

        Arguments:
          x: input tensor.
          filters: filters in `Conv2D`.
          num_row: height of the convolution kernel.
          num_col: width of the convolution kernel.
          padding: padding mode in `Conv2D`.
          strides: strides in `Conv2D`.
          name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

        Returns:
          Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
            bn_axis = 3
        x = layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(
            x)
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = layers.Activation('relu', name=name)(x)
        return x
    def __init__(self,input_shape,partition_layer,final_layer, **kwargs):
        super(InceptionWJ, self).__init__(**kwargs)
        channel_axis = 3
        img_input = layers.Input(shape=input_shape)
        # partition layer
        flag = False
        self.layer_list = []
        print("##########", img_input)

        if partition_layer == 'input':
            flag = True
            x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

        if flag or partition_layer == 'conv2d':
            if partition_layer == 'conv2d':
                x = img_input
                flag = True
            x = conv2d_bn(x, 32, 3, 3, padding='valid')

        if flag or partition_layer == 'conv2d_1':
            if partition_layer == 'conv2d_1':
                x = img_input
                flag = True
            x = conv2d_bn(x, 64, 3, 3)

        if flag or partition_layer == 'conv2d_2':
            if partition_layer == 'conv2d_2':
                x = img_input
                flag = True
            x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        if flag or partition_layer == 'max_pooling2d':
            if partition_layer == 'max_pooling2d':
                x = img_input
                flag = True
            x = conv2d_bn(x, 80, 1, 1, padding='valid')
        if flag or partition_layer == 'conv2d_3':
            if partition_layer == 'conv2d_3':
                x = img_input
                flag = True
            x = conv2d_bn(x, 192, 3, 3, padding='valid')
            print("==========partition_layer", partition_layer)
        if flag or partition_layer == 'conv2d_4':
            if partition_layer == 'conv2d_4':
                x = img_input
                flag = True
            x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        if flag or partition_layer == 'max_pooling2d_1':
            # print("=====================partition_layer", partition_layer)
            if partition_layer == 'max_pooling2d_1':
                x = img_input
                flag = True

            # mixed 0: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, 64, 1, 1)
            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
            x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed0')
        if flag or partition_layer == 'mixed0':

            if partition_layer == 'mixed0':
                x = img_input
                flag = True
            # mixed 1: 35 x 35 x 288
            branch1x1 = conv2d_bn(x, 64, 1, 1)

            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
            x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed1')
        if flag or partition_layer == 'mixed1':
            if partition_layer == 'mixed1':
                x = img_input
                flag = True
            # mixed 2: 35 x 35 x 288
            branch1x1 = conv2d_bn(x, 64, 1, 1)

            branch5x5 = conv2d_bn(x, 48, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
            x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed2')
        if flag or partition_layer == 'mixed2':
            if partition_layer == 'mixed2':
                x = img_input
                flag = True
            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

            branch3x3dbl = conv2d_bn(x, 64, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

            branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
            x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed3')
        if flag or partition_layer == 'mixed3':
            if partition_layer == 'mixed3':
                x = img_input
                flag = True
            # mixed 4: 17 x 17 x 768
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 128, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 128, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed4')

        # mixed 5, 6: 17 x 17 x 768

        if flag or partition_layer == 'mixed4':
            if partition_layer == 'mixed4':
                x = img_input
                flag = True
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(
                x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed' + str(5))

        if flag or partition_layer == 'mixed5':
            if partition_layer == 'mixed5':
                x = img_input
                flag = True
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(
                x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed' + str(6))
        if flag or partition_layer == 'mixed6':
            if partition_layer == 'mixed6':
                flag = True
                x = img_input
            # mixed 7: 17 x 17 x 768
            self.layer_list.append()
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 192, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 192, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed7')
        if flag or partition_layer == 'mixed7':
            if partition_layer == 'mixed7':
                x = img_input
                flag = True
            # mixed 8: 8 x 8 x 1280
            branch3x3 = conv2d_bn(x, 192, 1, 1)
            branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

            branch7x7x3 = conv2d_bn(x, 192, 1, 1)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
            branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
            branch7x7x3 = conv2d_bn(
                branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

            branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
            x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                                   axis=channel_axis,
                                   name='mixed8')

        # mixed 9: 8 x 8 x 2048
        i = 0
        if flag or partition_layer == 'mixed8':
            if partition_layer == 'mixed8':
                x = img_input
                flag = True
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                           axis=channel_axis,
                                           name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                              axis=channel_axis)

            branch_pool = layers.AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(
                x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed' + str(9 + i))
        i = 1
        if flag or partition_layer == 'mixed9':
            if partition_layer == 'mixed9':
                x = img_input
                flag = True
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                           axis=channel_axis,
                                           name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                              axis=channel_axis)

            branch_pool = layers.AveragePooling2D((3, 3),
                                                  strides=(1, 1),
                                                  padding='same')(
                x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                                   axis=channel_axis,
                                   name='mixed' + str(9 + i))
        if partition_layer == 'mixed10' or flag:
            if partition_layer == 'mixed10':
                x = img_input
                flag = True

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(1000, activation="softmax",
                         name='predictions')(x)
        #model = tf.keras.Model(img_input, x, name='inception_v3')
        return x

    def call(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
  """Utility function to apply conv + BN.

  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    num_row: height of the convolution kernel.
    num_col: width of the convolution kernel.
    padding: padding mode in `Conv2D`.
    strides: strides in `Conv2D`.
    name: name of the ops; will become `name + '_conv'`
      for the convolution and `name + '_bn'` for the
      batch norm layer.

  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
    bn_axis = 3
  x = layers.Conv2D(
      filters, (num_row, num_col),
      strides=strides,
      padding=padding,
      use_bias=False,
      name=conv_name)(
          x)
  x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = layers.Activation('relu', name=name)(x)
  return x

def inception_v3_wj(img_input,partition_layer):
    channel_axis = 3
    img_input = layers.Input(shape=img_input.shape[1:])
    # partition layer

    flag = False
    print("##########", img_input)

    if partition_layer == 'input':
        flag = True
        x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

    if flag or partition_layer == 'conv2d':
        if partition_layer == 'conv2d':
            x = img_input
            flag = True
        x = conv2d_bn(x, 32, 3, 3, padding='valid')

    if flag or partition_layer == 'conv2d_1':
        if partition_layer == 'conv2d_1':
            x = img_input
            flag = True
        x = conv2d_bn(x, 64, 3, 3)

    if flag or partition_layer == 'conv2d_2':
        if partition_layer == 'conv2d_2':
            x = img_input
            flag = True
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    if flag or partition_layer == 'max_pooling2d':
        if partition_layer == 'max_pooling2d':
            x = img_input
            flag = True
        x = conv2d_bn(x, 80, 1, 1, padding='valid')
    if flag or partition_layer == 'conv2d_3':
        if partition_layer == 'conv2d_3':
            x = img_input
            flag = True
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        print("==========partition_layer", partition_layer)
    if flag or partition_layer == 'conv2d_4':
        if partition_layer == 'conv2d_4':
            x = img_input
            flag = True
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    if flag or partition_layer == 'max_pooling2d_1':
        # print("=====================partition_layer", partition_layer)
        if partition_layer == 'max_pooling2d_1':
            x = img_input
            flag = True

        # mixed 0: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)
        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed0')
    if flag or partition_layer == 'mixed0':

        if partition_layer == 'mixed0':
            x = img_input
            flag = True
        # mixed 1: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed1')
    if flag or partition_layer == 'mixed1':
        if partition_layer == 'mixed1':
            x = img_input
            flag = True
        # mixed 2: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed2')
    if flag or partition_layer == 'mixed2':
        if partition_layer == 'mixed2':
            x = img_input
            flag = True
        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed3')
    if flag or partition_layer == 'mixed3':
        if partition_layer == 'mixed3':
            x = img_input
            flag = True
        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed4')

    # mixed 5, 6: 17 x 17 x 768

    if flag or partition_layer == 'mixed4':
        if partition_layer == 'mixed4':
            x = img_input
            flag = True
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(
            x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed' + str(5))

    if flag or partition_layer == 'mixed5':
        if partition_layer == 'mixed5':
            x = img_input
            flag = True
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(
            x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed' + str(6))
    if flag or partition_layer == 'mixed6':
        if partition_layer == 'mixed6':
            flag = True
            x = img_input
        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed7')
    if flag or partition_layer == 'mixed7':
        if partition_layer == 'mixed7':
            x = img_input
            flag = True
        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv2d_bn(x, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                               axis=channel_axis,
                               name='mixed8')

    # mixed 9: 8 x 8 x 2048
    i = 0
    if flag or partition_layer == 'mixed8':
        if partition_layer == 'mixed8':
            x = img_input
            flag = True
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                       axis=channel_axis,
                                       name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                          axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(
            x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed' + str(9 + i))
    i = 1
    if flag or partition_layer == 'mixed9':
        if partition_layer == 'mixed9':
            x = img_input
            flag = True
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                       axis=channel_axis,
                                       name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2],
                                          axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(
            x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed' + str(9 + i))
    if partition_layer == 'mixed10' or flag:
        if partition_layer == 'mixed10':
            x = img_input
            flag = True

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(1000, activation="softmax",
                     name='predictions')(x)
    #model = tf.keras.Model(img_input, x, name='inception_v3')
    return x
def incpetion_test():
    input_data = np.load("middle_data/inception_v3/Conv2d_2a_3x3_guitar.npy")
    input = tf.constant(np.asarray(np.expand_dims(input_data, axis=0),dtype=np.float32))
    layer_name_list =["input_1","conv2d","conv2d_1","conv2d_2","max_pooling2d","conv2d_3","conv2d_4","max_pooling2d_1"
                     ,"mixed0","mixed1","mixed2","mixed3","mixed4","mixed5","mixed6","mixed7","mixed8","mixed9","mixed10","predictions"]
    layer_name = 'conv2d'
    target_layer = inception_v3_wj(tf.constant(input),layer_name)
    print(target_layer)
input_basic = np.load("middle_data/inception_v3/input_guitar.npy")##
input_data = tf.constant(np.asarray(np.expand_dims(input_basic, axis=0),dtype=np.float32))

middle_basic = np.load("middle_data/inception_v3/Conv2d_1a_3x3_guitar.npy")
middle_data = tf.constant(np.asarray(np.expand_dims(middle_basic, axis=0),dtype=np.float32))
inception = inception_v3.InceptionV3(partition_layer='conv2d',input_shape=middle_basic.shape)
# for layer in inception.layers:
#    print(layer.name)

print(np.argmax(inception(middle_data)[0]))

'''
middle_basic = np.load("middle_data/inception_v3/Mixed_5b_guitar.npy")
middle_data = tf.constant(np.asarray(np.expand_dims(middle_basic, axis=0),dtype=np.float32))
layer_name = 'mixed0'
partition_layer = inception.get_layer(name=layer_name)
#print("++++++++",inception.inputs[0].extend(tf.keras.Input(middle_basic.shape)))
result = tf.keras.models.Model(inputs=[inception.inputs,tf.keras.Input(middle_basic.shape)],outputs=inception.output)
print(np.argmax(result(middle_data)[0]))
'''
