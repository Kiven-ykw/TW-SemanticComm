
"""
------------------------------------------------------------------------------------------------------
Simulation for the paper:

             Two-Way semantic communication without information feedback

Description:

The Transmitter and Receiver are trained iteration.

Author  : Kaiwen YU
Date    : 2022/11/10
------------------------------------------------------------------------------------------------------
"""

import glob
import sys
import keras
import math
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers, datasets
from channel import channel
#from torchinfo import summary
from utils import generate_ds, PeakSignalToNoiseRatio, StructuralSimilarityIndex
from model import semantic_autoencoder, generator_model, discirminator_model
from config import parse_args
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'


def discriminator_loss(real_output, fake_output):
    # Discriminator has to maximize the probability of assigning the correct label.
    # Real/Fake output is the probability of predicting the input as real.
    # real_output is the disc probability when real image is passed. Disc should pred it as close to 1.
    # fake_output is the disc probability when generated image is passed. Disc should pred it as close to 0.
    real_image_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
    fake_image_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
    return (real_image_loss + fake_image_loss)


def generator_loss(fake_output):
    # Generator has to fool to discriminator into predicting generated image as real.
    # fake_output is the disc probability when generated image is passed.
    # For generated to fool the disc, fake_output should be as close to 1 as poosible.
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output))


def main(args):

    # logs
    import datetime
    log_dir = "Twoway_SC/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    import os
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root=datasets.mnist,
                                    train_im_height=args.image_height,
                                    train_im_width=args.image_width,
                                    batch_size=args.batch_size,
                                    val_rate=args.val_rate,
                                    cache_data=True)

    # create model of SC-JPSC
    generator = generator_model()
    generator.build(input_shape=[(128, 256, args.noise_dim), (128, 256, 2)])

    generator.summary()

    discriminator = discirminator_model()
    discriminator.build(input_shape=[(128, 256, 2), (128, 256, 2)])
    discriminator.summary()

    Twoway_SC = semantic_autoencoder(parse_args())
    Twoway_SC.build(input_shape=[(1, 28, 28, 1), (1, 28, 28, 1)])
    Twoway_SC.summary()

    Tx_A = keras.Sequential([Twoway_SC.get_layer('SE_A'), Twoway_SC.get_layer('CE_A')])
    Rx_A = keras.Sequential([Twoway_SC.get_layer('CD_A'), Twoway_SC.get_layer('SD_A')])

    Tx_B = keras.Sequential([Twoway_SC.get_layer('SE_B'), Twoway_SC.get_layer('CE_B')])
    Rx_B = keras.Sequential([Twoway_SC.get_layer('CD_B'), Twoway_SC.get_layer('SD_B')])

    if not os.path.exists("Twoway_SC/save_weights"):
        # Create a folder to save weights
        os.makedirs("Twoway_SC/save_weights")

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / args.num_epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * args.lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)
        return new_lr

    # using keras low level api for training semantic channel
    loss_object = tf.keras.losses.MeanSquaredError()

    optimizer_gen = optimizers.Adam(learning_rate=args.lr)
    optimizer_disc = optimizers.Adam(learning_rate=args.lr)

    train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
    train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')

    val_loss_gen_channel_A = tf.keras.metrics.Mean(name='val_loss_gen_channel_A')
    val_psnr_gen_channel_A = PeakSignalToNoiseRatio(name='val_psnr_gen_channel_A')

    val_loss_gen_channel_B = tf.keras.metrics.Mean(name='val_loss_gen_channel_B')
    val_psnr_gen_channel_B = PeakSignalToNoiseRatio(name='val_psnr_gen_channel_B')

    val_loss_real_channel_A = tf.keras.metrics.Mean(name='val_loss_real_channel_A')
    val_psnr_real_channel_A = PeakSignalToNoiseRatio(name='val_psnr_real_channel_A')

    val_loss_real_channel_B = tf.keras.metrics.Mean(name='val_loss_real_channel_B')
    val_psnr_real_channel_B = PeakSignalToNoiseRatio(name='val_psnr_real_channel_B')

    # using keras low level api for training Tx
    loss_object_tx_A = tf.keras.losses.MeanSquaredError()
    optimizer_tx_A = optimizers.Adam(learning_rate=args.lr)

    train_loss_tx_A = tf.keras.metrics.Mean(name='train_loss_tx_A')
    train_psnr_tx_A = PeakSignalToNoiseRatio(name='train_psnr_tx_A')

    loss_object_tx_B = tf.keras.losses.MeanSquaredError()
    optimizer_tx_B = optimizers.Adam(learning_rate=args.lr)

    train_loss_tx_B = tf.keras.metrics.Mean(name='train_loss_tx_B')
    train_psnr_tx_B = PeakSignalToNoiseRatio(name='train_psnr_tx_B')

    # using keras low level api for training Rx
    loss_object_rx_A = tf.keras.losses.MeanSquaredError()
    optimizer_rx_A = optimizers.Adam(learning_rate=args.lr)

    train_loss_rx_A = tf.keras.metrics.Mean(name='train_loss_rx_A')
    train_psnr_rx_A = PeakSignalToNoiseRatio(name='train_psnr_rx_A')

    loss_object_rx_B = tf.keras.losses.MeanSquaredError()
    optimizer_rx_B = optimizers.Adam(learning_rate=args.lr)

    train_loss_rx_B = tf.keras.metrics.Mean(name='train_loss_rx_B')
    train_psnr_rx_B = PeakSignalToNoiseRatio(name='train_psnr_rx_B')

    @tf.function
    def train_channel_receiver_step(pilot_A, pilot_B):                          # We only modeling one channel since Channel reciprocity
        x_A = Tx_A(pilot_A, training=False)  # real data
        x_B = Tx_B(pilot_A, training=False)  # real data
        y_A = channel(x_A, args.snr_train_dB_up, 'AWGN')   # real channel
        y_B = channel(x_B, args.snr_train_dB_up, 'AWGN')  # real channel
        noise = tf.random.normal([x_A.shape[0], x_A.shape[1], args.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as rx_tape_A, tf.GradientTape() as rx_tape_B:
            gen_data = generator([noise, x_A])  # fake channel
            fake_data = discriminator([gen_data, x_A])
            real_data = discriminator([y_A, x_A])
            # losses
            disc_loss = discriminator_loss(real_data, fake_data)
            gen_loss = generator_loss(fake_data)

            real_receiver_A = Rx_A(y_A)
            real_receiver_B = Rx_B(y_B)
            # losses
            # rx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_receiver, labels=pilot))
            rx_loss_A = loss_object_rx_A(pilot_A, real_receiver_A)
            rx_loss_B = loss_object_rx_B(pilot_A, real_receiver_B)

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        dis_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        optimizer_gen.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        optimizer_disc.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
        # update loss and acc

        train_loss_disc(disc_loss)
        train_loss_gen(gen_loss)

        rx_gradients_A = rx_tape_A.gradient(rx_loss_A, Rx_A.trainable_variables)
        rx_gradients_B = rx_tape_B.gradient(rx_loss_B, Rx_B.trainable_variables)

        optimizer_rx_A.apply_gradients(zip(rx_gradients_A, Rx_A.trainable_variables))
        optimizer_rx_B.apply_gradients(zip(rx_gradients_B, Rx_B.trainable_variables))
        # update loss and acc

        train_loss_rx_A(rx_loss_A)
        train_loss_rx_B(rx_loss_B)
        train_psnr_rx_A(pilot_A, real_receiver_A)
        train_psnr_rx_B(pilot_A, real_receiver_B)

    # train_psnr(real_receiver, gen_receiver)

    @tf.function
    def train_transmitter_step(pilot_A, pilot_B):

        with tf.GradientTape() as tx_tape_A, tf.GradientTape() as tx_tape_B:
            x_A = Tx_A(pilot_A, training=True)
            x_B = Tx_B(pilot_A, training=True)
            noise = tf.random.normal([x_A.shape[0], x_A.shape[1], args.noise_dim])
            gen_data_A = generator([noise, x_A])  # fake channel
            gen_data_B = generator([noise, x_B])  # fake channel
            gen_receiver_A = Rx_B(gen_data_A)  # fake data   this is opposite
            gen_receiver_B = Rx_A(gen_data_B)  # fake data
            # losses
            #tx_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_receiver, labels=pilot))
            tx_loss_A = loss_object_tx_A(pilot_A, gen_receiver_A)
            tx_loss_B = loss_object_tx_B(pilot_A, gen_receiver_B)

        tx_gradients_A = tx_tape_A.gradient(tx_loss_A, Tx_A.trainable_variables)
        tx_gradients_B = tx_tape_B.gradient(tx_loss_B, Tx_B.trainable_variables)

        optimizer_tx_A.apply_gradients(zip(tx_gradients_A, Tx_A.trainable_variables))
        optimizer_tx_B.apply_gradients(zip(tx_gradients_B, Tx_B.trainable_variables))
        # update loss and acc

        train_loss_tx_A(tx_loss_A)
        train_loss_tx_B(tx_loss_B)
        train_psnr_tx_A(pilot_A, gen_receiver_A)
        train_psnr_tx_B(pilot_A, gen_receiver_B)

    @tf.function
    def val_step(val_A, val_B):
        x_A = Tx_A(val_A, training=False)              # real data
        x_B = Tx_B(val_B, training=False)  # real data
        y_A = channel(x_A, args.snr_train_dB_up, 'AWGN')   # real channel
        y_B = channel(x_B, args.snr_train_dB_up, 'AWGN')  # real channel
        real_image_A = Rx_A(y_A)
        real_image_B = Rx_B(y_B)

        noise = tf.random.normal([x_A.shape[0], x_A.shape[1], args.noise_dim])   # fake data
        gen_data_A = generator([noise, x_A])                           # fake data
        gen_data_B = generator([noise, x_B])  # fake data
        fake_image_A = Rx_A(gen_data_A)
        fake_image_B = Rx_B(gen_data_B)

        loss_real_channel_A = loss_object(real_image_A, val_A)
        loss_gen_channel_A = loss_object(fake_image_A, val_A)

        loss_real_channel_B = loss_object(real_image_B, val_B)
        loss_gen_channel_B = loss_object(fake_image_B, val_B)

        val_loss_real_channel_A(loss_real_channel_A)
        val_psnr_real_channel_A(real_image_A, val_A)

        val_loss_gen_channel_A(loss_gen_channel_A)
        val_psnr_gen_channel_A(fake_image_A, val_A)

        val_loss_real_channel_B(loss_real_channel_B)
        val_psnr_real_channel_B(real_image_B, val_B)

        val_loss_gen_channel_B(loss_gen_channel_B)
        val_psnr_gen_channel_B(fake_image_B, val_B)

    best_val_loss = 1.
    for epoch in range(args.num_epochs):
        # clear train history info
        train_loss_gen.reset_states()
        train_loss_disc.reset_states()
        train_loss_tx_A.reset_state()
        train_loss_rx_A.reset_state()
        train_loss_tx_B.reset_state()
        train_loss_rx_B.reset_state()
        train_psnr_tx_A.reset_state()
        train_psnr_tx_B.reset_state()
        # clear val history info
        val_loss_real_channel_A.reset_states()
        val_psnr_real_channel_A.reset_states()
        val_loss_gen_channel_A.reset_states()
        val_psnr_gen_channel_A.reset_states()
        val_loss_real_channel_B.reset_states()
        val_psnr_real_channel_B.reset_states()
        val_loss_gen_channel_B.reset_states()
        val_psnr_gen_channel_B.reset_states()
        # train
        print("train epoch [{}/{}]".format(epoch+1, args.num_epochs))
        number_steps_channel = args.number_steps_channel
        for step in range(number_steps_channel):
            train_bar = tqdm(train_ds, file=sys.stdout)
            for images_A, images_B in train_bar:
                train_channel_receiver_step(images_A, images_B)
                # print train process
                train_bar.desc = "Epoch [{}]: train channel step[{}/{}] loss_disc:{:.4f}, loss_gen:{:.4f}, " \
                                 "loss_rx_A:{:.4f}, psnr_rx_A:{:.4f}; loss_rx_B:{:.4f}, psnr_rx_B:{:.4f}".format(
                                    epoch + 1,
                                    step + 1,
                                    number_steps_channel,
                                    train_loss_disc.result(),
                                    train_loss_gen.result(),
                                    train_loss_rx_A.result(),
                                    train_psnr_rx_A.result(),
                                    train_loss_rx_B.result(),
                                    train_psnr_rx_B.result(),
                                    )
            # update learning rate
            if args.lr_decay:
                optimizer_gen.learning_rate = scheduler(step)
                optimizer_disc.learning_rate = scheduler(step)

        number_steps_transmitter = args.number_steps_transmitter
        for step in range(number_steps_transmitter):
            train_bar = tqdm(train_ds, file=sys.stdout)
            for images_A, images_B in train_bar:
                train_transmitter_step(images_A, images_B)
                # print train process
                train_bar.desc = "Epoch [{}]: train transmitter step[{}/{}] loss_tx_A:{:.4f}, psnr_tx_A:{:.4f}; loss_tx_B:{:.4f}, psnr_tx_B:{:.4f}".format(
                    epoch + 1,
                    step + 1,
                    number_steps_transmitter,
                    train_loss_tx_A.result(),
                    train_psnr_tx_A.result(),
                    train_loss_tx_B.result(),
                    train_psnr_tx_B.result(),
                )
            # update learning rate
            if args.lr_decay:
                optimizer_tx_A.learning_rate = scheduler(step)
                optimizer_tx_B.learning_rate = scheduler(step)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images_A, images_B in val_bar:
            val_step(images_A, images_A)
            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss_real_channel_A:{:.4f}, loss_gen_channel_A:{:.4f}, " \
                           "loss_real_channel_B:{:.4f}, loss_gen_channel_B:{:.4f}" \
                           "psnr_real_channel_A:{:.4f}, psnr_gen_channel_A:{:.4f};" \
                           "psnr_real_channel_B:{:.4f}, psnr_gen_channel_B:{:.4f};".format(
                            epoch + 1,
                            args.num_epochs,
                            val_loss_real_channel_A.result(),
                            val_loss_gen_channel_A.result(),
                            val_loss_real_channel_B.result(),
                            val_loss_gen_channel_B.result(),
                            val_psnr_real_channel_A.result(),
                            val_psnr_gen_channel_A.result(),
                            val_psnr_real_channel_B.result(),
                            val_psnr_gen_channel_B.result(),
                            )

        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss_disc", train_loss_disc.result(), epoch)
            tf.summary.scalar("loss_gen", train_loss_gen.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss_real_channel_A.result(), epoch)
            tf.summary.scalar("ssim_A", val_psnr_real_channel_A.result(), epoch)
            tf.summary.scalar("ssim_B", val_psnr_real_channel_B.result(), epoch)

        # only save best weights
        if val_loss_real_channel_A.result() + val_loss_real_channel_B.result() < best_val_loss:
            best_val_loss = val_loss_real_channel_A.result() + val_loss_real_channel_B.result()
            save_name_gen = "Twoway_SC/save_weights/Gen_pilot_awgn_7db1_mse.ckpt"
            save_name_disc = "Twoway_SC/save_weights/Disc_pilot_awgn_7db1_mse.ckpt"
            save_name_tx_A = "Twoway_SC/save_weights/Tx_A_pilot_awgn_7db1_mse.ckpt"
            save_name_rx_A = "Twoway_SC/save_weights/Rx_A_pilot_awgn_7db1_mse.ckpt"
            save_name_tx_B = "Twoway_SC/save_weights/Tx_B_pilot_awgn_7db1_mse.ckpt"
            save_name_rx_B = "Twoway_SC/save_weights/Rx_B_pilot_awgn_7db1_mse.ckpt"
            generator.save_weights(save_name_gen, save_format="tf")
            discriminator.save_weights(save_name_disc, save_format="tf")
            Tx_A.save_weights(save_name_tx_A, save_format="tf")
            Rx_A.save_weights(save_name_rx_A, save_format="tf")
            Tx_B.save_weights(save_name_tx_B, save_format="tf")
            Rx_B.save_weights(save_name_rx_B, save_format="tf")
            print('save model!')


if __name__ == '__main__':
    args = parse_args()
    print("Called with args:", args)
    main(args)