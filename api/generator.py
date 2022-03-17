from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow as tf


def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization())

    result.add(layers.PReLU())

    return result


def upsample(filters, size, apply_batch=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if apply_batch:
        # tfa.layers.InstanceNormalization())
        result.add(layers.BatchNormalization())

    result.add(layers.PReLU())

    return result


class self_attention(tf.keras.Model):
    def __init__(self, channels):
        super(self_attention, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2D(channels // 8, kernel_size=1,
                               strides=1, )  # [bs, h, w, c']
        self.g = layers.Conv2D(channels // 8, kernel_size=1,
                               strides=1, )  # [bs, h, w, c']
        self.h = layers.Conv2D(channels // 2, kernel_size=1,
                               strides=1, )  # [bs, h, w, c]
        self.last_ = layers.Conv2D(
            self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def hw_flatten(self, x):
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(
            f), transpose_b=True)/tf.math.sqrt(dk)  # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h), transpose_a=True)  # [bs, N, C]
        # tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        gamma = 0.002
        o = self.reshape(o, height, width, num_channels)  # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)

        return out


class self_attentionDecoder(tf.keras.Model):
    def __init__(self, channels):
        super(self_attentionDecoder, self).__init__(name='')
        self.channels = channels
        self.f = layers.Conv2DTranspose(
            channels // 8, kernel_size=1, strides=1, )  # [bs, h, w, c']
        self.g = layers.Conv2DTranspose(
            channels // 8, kernel_size=1, strides=1, )  # [bs, h, w, c']
        self.h = layers.Conv2DTranspose(
            channels // 2, kernel_size=1, strides=1, )  # [bs, h, w, c]
        self.last_ = layers.Conv2DTranspose(
            self.channels, kernel_size=1, strides=1, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def hw_flatten(self, x):
        # layers.Reshape(( -1, x.shape[-2], x.shape[-1]))(x)
        return layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    def reshape(self, x, height, width, num_channels):
        return layers.Reshape((height, width, num_channels//2))(x)

    def call(self, x):
        batch_size, height, width, num_channels = x.get_shape().as_list()

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)
        dk = tf.cast(tf.shape(g)[-1], tf.float32)

        # N = h * w
        s = tf.matmul(self.hw_flatten(g), self.hw_flatten(
            f), transpose_b=True)/tf.math.sqrt(dk)  # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, self.hw_flatten(h))  # [bs, N, C]
        # tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        gamma = 0.002
        o = self.reshape(o, height, width, num_channels)  # [bs, h, w, C]
        o = self.last_(o)
        out = self.dropout(o)
        out = self.layernorm(gamma * out + x)

        return out


def Generator(HEIGHT, WIDTH):
    inputs = layers.Input(shape=[HEIGHT, WIDTH, 3])
    OUTPUT_CHANNELS = 3
    # bs = batch size
    down_stack = [
        downsample(64, 4, apply_instancenorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)

        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4)  # (bs, 16, 16, 512)
    ]

    up_stack = [
        upsample(512, 4, True),  # (bs, 16, 16, 1024)
        upsample(512, 4, True),  # (bs, 16, 16, 1024)

        upsample(512, 4, True),  # (bs, 16, 16, 1024)
        upsample(256, 4, True),  # (bs, 32, 32, 512)
        upsample(128, 4, True),  # (bs, 64, 64, 256)
        upsample(64, 4, True)  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # Bottleneck
    attention0 = self_attention(512)
    x = attention0(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    attention1 = self_attentionDecoder(192)
    x = attention1(x)
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)
