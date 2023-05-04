import math
from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf
from config import parse_args
import glob


# ResBlk -> Tx
class ResidualBlockTx(layers.Layer):

    def __init__(self, out_channel, strides: int=1, downsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:

            self.Conv2D_0 = layers.Conv2D(filters=self.out_channel, kernel_size=1, 
                                            strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv2D_1 = layers.Conv2D(filters=self.out_channel, kernel_size=3, 
                                            strides=self.strides, padding='same', activation=None)
        self.Conv2D_2 = layers.Conv2D(filters=self.out_channel, kernel_size=3, 
                                            strides=1, padding='same',activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv2D_0(inputs)
            residual = self.BatchNorm_0(residual)
        x = self.Conv2D_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv2D_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['downsample'] = self.downsample

        return base_config


# ResBlk -> R
class ResidualBlock(layers.Layer):

    def __init__(self, out_channel, strides: int=1, downsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv1D_0 = layers.Conv1D(filters=self.out_channel, kernel_size=1, 
                                            strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv1D_1 = layers.Conv1D(filters=self.out_channel, kernel_size=3, 
                                        strides=self.strides, padding='same', activation=None)
        self.Conv1D_2 = layers.Conv1D(filters=self.out_channel, kernel_size=3, 
                                        strides=1, padding='same', activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv1D_0(inputs)
            residual = self.BatchNorm_0(residual)
        x = self.Conv1D_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv1D_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['downsample'] = self.downsample

        return base_config


# ResBlk -> Rx
class ResidualBlockRx(layers.Layer):

    def __init__(self, out_channel, strides: int=1, upsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.upsample = upsample
        if self.upsample:
            self.Conv2DTrans_0 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=1, 
                                                        strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv2DTrans_1 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3, 
                                                        strides=self.strides, padding='same', activation=None)
        self.Conv2DTrans_2 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3,
                                                        strides=1, padding='same', activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.upsample:
            residual = self.Conv2DTrans_0(inputs)
            residual = self.BatchNorm_0(residual)
        x = self.Conv2DTrans_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv2DTrans_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['upsample'] = self.upsample

        return base_config


class generator_conditional(layers.Layer):
    def __init__(self, name: str=None):
        super(generator_conditional, self).__init__(name=None)
        self.Conv2D_1 = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='leaky_relu')
        self.Conv2D_2 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='leaky_relu')
        self.Conv2D_3 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='leaky_relu')
        self.Conv2D_4 = layers.Conv1D(filters=2, kernel_size=3, padding='same')

    def call(self, inputs):
        z = inputs[0]
        conditioning = inputs[1]
        z_conbine = tf.concat([z, conditioning], -1)
        outputs = self.Conv2D_1(z_conbine)
        outputs = self.Conv2D_2(outputs)
        outputs = self.Conv2D_3(outputs)
        outputs = self.Conv2D_4(outputs)

        return outputs


class discriminator_conditional(layers.Layer):
    def __init__(self, name: str=None):
        super(discriminator_conditional, self).__init__(name=None)
        self.Conv2D_1 = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')
        self.Conv2D_2 = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.Conv2D_3 = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.Conv2D_4 = layers.Conv1D(filters=16, kernel_size=3, padding='same')
        self.Flatten = layers.Flatten()
        self.Dence = layers.Dense(units=100, activation=None)
        #self.FC = tf.nn.relu(layers.Dense(units=100, activation=None))
        self.D_logit = layers.Dense(units=1, activation=None)
        #self.D_prob = tf.nn.sigmoid()

    def call(self, inputs):
        x = inputs[0]
        conditioning = inputs[1]
        z_conbine = tf.concat([x, conditioning], -1)
        z = self.Conv2D_1(z_conbine)
        z = tf.reduce_mean(z, axis=0, keepdims=True)
        z = self.Conv2D_2(z)
        z = self.Conv2D_3(z)
        z = self.Conv2D_4(z)
        z = self.Flatten(z)
        z = self.Dence(z)
        z = tf.nn.relu(z)
        D_logit = self.D_logit(z)
       #D_prob = tf.nn.sigmoid(D_logit)

        return D_logit#, D_prob

# semantic encoder
class SemanticEncoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(SemanticEncoder, self).__init__(name=name, **kwargs)
        self.Conv2D = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same', activation='elu')
        self.ResBlk_1 = ResidualBlockTx(out_channel=8, strides=2, downsample=True)
        self.ResBlk_2 = ResidualBlockTx(out_channel=16, strides=2, downsample=True)

    def call(self, inputs, *args, **kwargs):
        x = self.Conv2D(inputs)
        x = self.ResBlk_1(x)
        x = self.ResBlk_2(x)

        return x


# channel encoder
class ChannelEncoder(layers.Layer):

    def __init__(self, num_symbol, name=None, **kwargs):
        super(ChannelEncoder, self).__init__(name=name, **kwargs)
        self.ResBlk_1 = ResidualBlockTx(out_channel=32, strides=1, downsample=True)
        self.ResBlk_2 = ResidualBlockTx(out_channel=32, strides=1, downsample=False)
        self.Flatten = layers.Flatten()
        self.Dense = layers.Dense(units=2*num_symbol, activation=None, use_bias=True)
        self.Reshape = layers.Reshape((-1, 2))

    def call(self, inputs, *args, **kwargs):
        x = self.ResBlk_1(inputs)
        x = self.ResBlk_2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Reshape(x)
        # power norm
        x_norm = tf.math.sqrt(tf.cast(x.shape[1], tf.float32) / 2.0) * tf.math.l2_normalize(x, axis=1)

        return x_norm


class ChannelLayer(layers.Layer):
    def __init__(self, snr_db, channel_type: str='AWGN', name: str=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.snr_db = snr_db
        self.channel_type = channel_type

    def call(self, inputs, *args, **kwargs):
        # noise std
        Es = 1
        EsN0 = 10 ** (self.snr_db / 10)
        N0 = Es / EsN0
        sigma = math.sqrt(N0 / 2)
        std = tf.constant(value=sigma, dtype=tf.float32)
        # signal
        inputs_real = inputs[:, :, 0]
        inputs_imag = inputs[:, :, 1]
        inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
        # AWGN channel
        if self.channel_type == 'AWGN':
            h_complex = tf.complex(real=1., imag=0.)
        # Rayleigh channel
        elif self.channel_type == 'Rayleigh':
            h_real = tf.divide(
                tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=1.0, dtype=tf.float32),
                tf.sqrt(2.))
            h_imag = tf.divide(
                tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=1.0, dtype=tf.float32),
                tf.sqrt(2.))
            h_complex = tf.complex(real=h_real, imag=h_imag)
        # noise
        n_real = tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=std, dtype=tf.float32)
        n_imag = tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=std, dtype=tf.float32)
        noise = tf.complex(real=n_real, imag=n_imag)
        # received signal y
        hx = tf.multiply(h_complex, inputs_complex)
        y_complex = tf.add(hx, noise)
        # reshape

        x_hat_complex = tf.math.divide_no_nan(y_complex, h_complex)
        x_hat_real = tf.math.real(x_hat_complex)
        x_hat_imag = tf.math.imag(x_hat_complex)
        x_hat_real = tf.expand_dims(x_hat_real, axis=-1)
        x_hat_imag = tf.expand_dims(x_hat_imag, axis=-1)
        x_hat = tf.concat([x_hat_real, x_hat_imag], axis=-1)

        return x_hat

    def get_config(self):
        base_config = super().get_config()
        base_config['snr_db'] = self.snr_db
        base_config['channel_type'] = self.channel_type

        return base_config


# channel decoder
class ChannelDecoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(ChannelDecoder, self).__init__(name=name, **kwargs)
        self.Concatenate = layers.Concatenate(axis=1)
        self.Flatten = layers.Flatten()
        self.Dense_1 = layers.Dense(units=7 * 7 * 16, activation=None, use_bias=True)
        self.Reshape = layers.Reshape((7, 7, 16))
        self.ResBlk_1 = ResidualBlockRx(out_channel=32, strides=1, upsample=True)
        self.ResBlk_2 = ResidualBlockRx(out_channel=16, strides=1, upsample=True)

    def call(self, inputs, *args, **kwargs):
        #oncat = self.Concatenate(inputs)
        x = self.Flatten(inputs)
        x = self.Dense_1(x)
        # x = self.Dense_2(x)
        x = self.Reshape(x)
        x = self.ResBlk_1(x)
        x = self.ResBlk_2(x)

        return x


# semantic decoder
class SemanticDecoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(SemanticDecoder, self).__init__(name=name, **kwargs)

        self.ResBlk_1 = ResidualBlockRx(out_channel=8, strides=2, upsample=True)
        self.ResBlk_2 = ResidualBlockRx(out_channel=4, strides=2, upsample=True)
        self.Flatten = layers.Flatten()
        self.TransConv2D = layers.Conv2DTranspose(filters=1, kernel_size=3, 
                                                    strides=1, padding='same', activation=None)
        self.Sigmoid = layers.Activation('sigmoid')
        self.drop = layers.Activation('Softmax')

    def call(self, inputs, *args, **kwargs):

        x = self.ResBlk_1(inputs)
        x = self.ResBlk_2(x)
        x = self.TransConv2D(x)
        x = self.Sigmoid(x)

        return x


# class SemanticTransmitter(Model):
#     def __init__(self, args):
#         super(SemanticTransmitter, self).__init__()
#
#         self.SE = SemanticEncoder(name='SE')
#
#         self.CE = ChannelEncoder(num_symbol=args.num_symbol_node, name='CE')
#
#     def call(self, inputs):
#         x = self.SE(inputs[0])
#         x = self.CE(x)
#         return x
#
#
# class SemanticReceiver(Model):
#     def __init__(self, args):
#         super(SemanticReceiver, self).__init__()
#
#         self.SD = SemanticDecoder(name='SD')
#
#         self.CD = ChannelDecoder(name='CD')
#
#     def call(self, y):
#         rec = self.SD(y)
#         rec = self.CD(rec)
#
#         return rec


class SemanticComm(Model):
    def __init__(self, args, train_channel='real_channel', **kwargs):
        super().__init__(**kwargs)

        self.train_channel = train_channel

        # transmitter
        self.SE_A = SemanticEncoder(name='SE_A')
        self.CE_A = ChannelEncoder(num_symbol=args.num_symbol_node, name='CE_A')
        self.SE_B = SemanticEncoder(name='SE_B')
        self.CE_B = ChannelEncoder(num_symbol=args.num_symbol_node, name='CE_B')

        if train_channel == 'real_channel':

            self.channel_A = ChannelLayer(snr_db=args.snr_train_dB_down, channel_type=args.channel_type, name='channel_A')
            self.channel_B = ChannelLayer(snr_db=args.snr_train_dB_down, channel_type=args.channel_type,
                                          name='channel_B')

        # receiver
        self.SD_A = SemanticDecoder(name='SD_A')
        self.CD_A = ChannelDecoder(name='CD_A')
        self.SD_B = SemanticDecoder(name='SD_B')
        self.CD_B = ChannelDecoder(name='CD_B')

    # Tx_A
    def transmitter_A(self, x):
        x = self.SE_A(x)
        x = self.CE_A(x)
        return x

    # Tx_B
    def transmitter_B(self, x):
        x = self.SE_B(x)
        x = self.CE_B(x)
        return x

    # Rx_A
    def receiver_A(self, x):
        x = self.CD_A(x)
        x = self.SD_A(x)
        return x

    # Rx_B
    def receiver_B(self, x):
        x = self.CD_B(x)
        x = self.SD_B(x)
        return x

    def call(self, inputs, training=None):

        x_A = self.transmitter_A(inputs[0])
        x_B = self.transmitter_B(inputs[1])

        y_A = self.channel_A(x_A)
        y_B = self.channel_B(x_B)

        y_A = self.receiver_A(y_A)
        y_B = self.receiver_B(y_B)

        return y_A, y_B


class _Generator_(Model):
    def __init__(self):
        super().__init__()
        self.gen_A = generator_conditional(name='gen_A')

    def call(self, inputs):
        rec_A = self.gen_A(inputs)
        return rec_A


class _Discriminator_(Model):
    def __init__(self):
        super().__init__()
        self.disc_A = discriminator_conditional(name='disc_A')

    def call(self, inputs):
        rec_A = self.disc_A(inputs)
        return rec_A


def semantic_autoencoder(args, **kwargs):
    model = SemanticComm(args)
    return model


def generator_model():
    model = _Generator_()
    return model


def discirminator_model():
    model = _Discriminator_()
    return model
