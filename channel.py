

import math
import tensorflow as tf


def channel(inputs, snr_db, channel_type: str='AWGN', *args, **kwargs):
    Es = 1
    EsN0 = 10 ** (snr_db / 10)
    N0 = Es / EsN0
    sigma = math.sqrt(N0 / 2)
    std = tf.constant(value=sigma, dtype=tf.float32)
    inputs_real = inputs[:, :,0]
    inputs_imag = inputs[:, :,1]
    inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
    # AWGN channel
    if channel_type == 'AWGN':
        h_complex = tf.complex(real=1., imag=0.)
    # Rayleigh channel
    elif channel_type == 'Rayleigh':
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

#
# if __name__ == '__main__':

