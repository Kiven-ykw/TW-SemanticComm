# kaiwen yu writing in 2022/11/8
# some part reference from github

import copy
import glob
import random
import scipy.io as scio
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from model import semantic_autoencoder as creat_model
from channel import channel
from config import parse_args
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.rcParams['font.serif'] = ['Times New Roman']

def data_loader():
    """
    load datasets
    """
    # Step1: load data mnist
    mnist = tf.keras.datasets.mnist
    _, (x_test, _) = mnist.load_data()
    # Step2: normalize
    x_test_ = x_test.astype('float32') / 255.
    x_test_ = x_test_.reshape(x_test.shape[0], 28, 28, 1)
    x_test_A = copy.deepcopy(x_test_[:5000])
    x_test_B = copy.deepcopy(x_test_[:5000])
    # Step3: shuffle
    random.shuffle(x_test_A)
    random.shuffle(x_test_B)

    return (x_test_A, x_test_B)


# 显示图片结果
def disp_result(decode, range_SNR, interval):
    """
    visualization
    :param decode: image data
    :param range_SNR: test the range of SNR
    :param interval: Relative Phase Offset value interval
    :return:
    """
    # SNR interval
    pl_rangeSNR = [x for x in range_SNR if x % interval == 0]
    # image size
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    for i in range(len(pl_rangeSNR)):
        # original_A -> first row
        p1 = plt.subplot(4, len(pl_rangeSNR), i + 1)
        p1.imshow(decode['Source_A'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p1.set_title('Source_A', fontproperties="SimHei", fontsize=10)
        # trans_A -> second row
        p2 = plt.subplot(4, len(pl_rangeSNR), i + len(pl_rangeSNR) + 1)
        p2.imshow(decode['Destination_A'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p2.set_title('SNR = ' + str(pl_rangeSNR[i]), fontproperties="SimHei", fontsize=10)

        # original_A -> first row
        p3 = plt.subplot(4, len(pl_rangeSNR), i + 1)
        p3.imshow(decode['Source_B'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p3.set_title('Source_B', fontproperties="SimHei", fontsize=10)
        # trans_A -> second row
        p4 = plt.subplot(4, len(pl_rangeSNR), i + len(pl_rangeSNR) + 1)
        p4.imshow(decode['Destination_B'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p4.set_title('SNR = ' + str(pl_rangeSNR[i]), fontproperties="SimHei", fontsize=10)

    plt.savefig('Twoway_SC/results/picture_predict_PSNR_AWGN.png', dpi=200)
    plt.close()
    # plt.show()


def plot_model_performance(psnr, range_SNR, name):
    """
    plot PSNR performance
    :param psnr_A: PSNR performance sent by node A to node B
    :param psnr_B: PSNR performance sent by node B to node A
    :param psnr: the average of psnr_A and psnr_B
    :param range_phase: Test the range of Relative Phase Offset
    :param name: name of save image
    :return:
    """
    plt.title('The effect of different SNR on PSNR')

    plt.plot(range_SNR,
             psnr,
             label='AWGN',
             linewidth='1',  # 粗细
             color='g',  # 颜色
             linestyle='-',  # 线型（line_style 简写为 ls）
             marker='.'  # 点型（标记marker）
             )
    plt.legend(fontsize=8, loc='lower right')
    plt.ylabel('Peak signal-to-noise ratio (dB)')
    plt.xlabel('SNR(dB)')
    plt.ylim([10, 40])
    plt.grid(linestyle='-.')
    file_dir = 'Twoway_SC/results/' + name + '.png'
    plt.savefig(file_dir, dpi=200)
    plt.close()
    # plt.show()


def sc_model():

    Twoway_SC = creat_model(parse_args())
    Twoway_SC.build(input_shape=[(1,28,28,1),(1,28,28,1)])
    Twoway_SC.summary()


    Tx_A = keras.Sequential([Twoway_SC.get_layer('SE_A'), Twoway_SC.get_layer('CE_A')])

    Rx_A = keras.Sequential([Twoway_SC.get_layer('CD_A'), Twoway_SC.get_layer('SD_A')])

    Tx_B = keras.Sequential([Twoway_SC.get_layer('SE_B'), Twoway_SC.get_layer('CE_B')])

    Rx_B = keras.Sequential([Twoway_SC.get_layer('CD_B'), Twoway_SC.get_layer('SD_B')])

    # load weight for Rx
    weights_path = 'Twoway_SC/save_weights/Rx_A_pilot_awgn_sp_mse.ckpt'
    assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
    Rx_A.load_weights(weights_path)

    weights_path = 'Twoway_SC/save_weights/Rx_B_pilot_awgn_sp_mse.ckpt'
    assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
    Rx_B.load_weights(weights_path)

    # load weight for Tx
    weights_path = 'Twoway_SC/save_weights/Tx_A_pilot_awgn_sp_mse.ckpt'
    assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
    Tx_A.load_weights(weights_path)

    weights_path = 'Twoway_SC/save_weights/Tx_B_pilot_awgn_sp_mse.ckpt'
    assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
    Tx_B.load_weights(weights_path)

    return Tx_A, Rx_A, Tx_B, Rx_B


# 预测模型
def predict_model():
    if not os.path.exists("Twoway_SC/results"):
        # Create a folder to save results
        os.makedirs("Twoway_SC/results")

    batch_size = 128
    # load model
    Tx_A, Rx_A, Tx_B, Rx_B = sc_model()
    (x_test_1, x_test_2) = data_loader()
    # SNR range
    SNR = np.arange(-10, 20, 4)
    # peak signal-to-noise ratio
    PSNR = []
    SSIM = []

    # save image to visualization
    x_predict = {'Source_A': [], 'Destination_A': [], 'Source_B': [], 'Destination_B': []}
    for SNR_dB in SNR:
        # randomly choose a location for comparison
        position_compare = random.randint(0, x_test_1.shape[0] - 1)
        print('SNR_dB=' + str(SNR_dB))

        # Step2: Predict
        x_A = Tx_A.predict(x=x_test_1, batch_size=batch_size)
        x_B = Tx_B.predict(x=x_test_2, batch_size=batch_size)

        y_A = channel(inputs=x_A, snr_db=SNR_dB, channel_type='AWGN', modulation='QPSK')
        y_B = channel(inputs=x_B, snr_db=SNR_dB, channel_type='AWGN', modulation='QPSK')

        b_A = Rx_A.predict(x=y_A, batch_size=batch_size)
        b_B = Rx_B.predict(x=y_B, batch_size=batch_size)

        # Step3: Results
        if SNR_dB % 4 == 0:
            x_predict['Source_A'].append(x_test_1[position_compare].copy())
            x_predict['Destination_A'].append(b_A[position_compare].copy())
            x_predict['Source_B'].append(x_test_2[position_compare].copy())
            x_predict['Destination_B'].append(b_B[position_compare].copy())
        psnr = tf.reduce_mean([tf.image.psnr(b_A, x_test_1, max_val=1), tf.image.psnr(b_B, x_test_2, max_val=1)])
        ssim = tf.reduce_mean([tf.image.ssim(b_A, x_test_1, 1), tf.image.ssim(b_B, x_test_2, 1)])
        PSNR.append(tf.reduce_mean(psnr).numpy())
        SSIM.append(tf.reduce_mean(ssim).numpy())

        print('PSNR_A=' + str(tf.reduce_mean(tf.image.psnr(b_A, x_test_1, max_val=1)).numpy()))
        print('PSNR_B=' + str(tf.reduce_mean(tf.image.psnr(b_B, x_test_2, max_val=1)).numpy()))

        print('PSNR='+str(tf.reduce_mean(psnr).numpy()))
        print('SSIM=' + str(tf.reduce_mean(ssim).numpy()))
    plot_model_performance(PSNR, SNR, name='PSNR_SNR')
    plot_model_performance(SSIM, SNR, name='SSIM_SNR')
    disp_result(x_predict, SNR, interval=4)

   # # save data
   #  if not os.path.exists("Twoway_SC/data"):
   #      # Create a folder to save results
   #      os.makedirs("Twoway_SC/data")
   #  dataNew = 'Twoway_SC/data/fd_gen_awgn_sp_test_awgn_mse.mat'
   #  scio.savemat(dataNew, {'SNR': list(SNR), 'PSNR': PSNR, 'SSIM': SSIM})


# 测试模型准确率
def evaluate_model():
    # 加载测试集
    pass


if __name__ == '__main__':
    predict_model()
