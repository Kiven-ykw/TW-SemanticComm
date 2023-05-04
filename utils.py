"""
DNN utils
"""

import tensorflow as tf


# use to mnist cifar
def generate_ds(data_root: tf.keras.datasets = tf.keras.datasets.cifar10,
                train_im_height: int = 28,
                train_im_width: int = 28,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                val_rate: float = 0.5,
                cache_data: bool = False):
    """
    :param data_root: 数据根目录
    :param train_im_height: 训练图像的高度
    :param train_im_width: 训练图像的宽度
    :param val_im_height: 验证图像的高度
    :param val_im_width: 验证图像的宽度
    :param batch_size: 训练使用的batch size
    :param val_rate: 将数据按给定比例划分到验证集
    :param cache_data: 是否缓存数据
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def process_train_info(img_path, train_ratio):
        # load train data from part of train datasets
        train_num = int(len(img_path)*train_ratio)
        image = img_path[:train_num]
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(input=image, axis=-1)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image=image, target_height=train_im_height, target_width=train_im_width)
        image = tf.image.random_flip_left_right(image=image)
        image = image / 255.
        return image

    def process_val_info(img_path, val_ratio):
        # load val datasets from part of test data
        val_num = int(len(img_path)*(1-val_ratio))
        image = img_path[val_num:]
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(input=image, axis=-1)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, val_im_height, val_im_width)
        image = image / 255.
        return image

    # Configure dataset for performance
    def configure_for_performance(ds, shuffle_size: int, shuffle: bool = False, cache: bool = False):
        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    (x_train_, _), (x_test_, _) = data_root.load_data()
    x_train_A = tf.random.shuffle(x_train_)
    x_train_B = tf.random.shuffle(x_train_)
    x_test_A = tf.random.shuffle(x_test_)
    x_test_B = tf.random.shuffle(x_test_)
    # load datasets
    x_train_A = process_train_info(img_path=x_train_A, train_ratio=1)
    x_train_B = process_train_info(img_path=x_train_B, train_ratio=1)
    x_val_A = process_val_info(img_path=x_test_A, val_ratio=val_rate)
    x_val_B = process_val_info(img_path=x_test_B, val_ratio=val_rate)

    # train datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train_A, x_train_B))
    total_train = len(x_train_A)
    print('train datasets:' + str(total_train))
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)

    # val datasets
    val_ds = tf.data.Dataset.from_tensor_slices((x_val_A, x_val_B))
    total_val = len(x_val_A)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)
    print('val datasets:' + str(total_val))
    
    return train_ds, val_ds


class PeakSignalToNoiseRatio(tf.keras.metrics.Metric):
    def __init__(self, name="peak_signal-to-noise_ratio", **kwargs):
        super(PeakSignalToNoiseRatio, self).__init__(name=name, **kwargs)
        self.PSNR = self.add_weight(name="PSNR", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        PSNR = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            PSNR = tf.multiply(PSNR, sample_weight)
        self.PSNR.assign(PSNR)

    def result(self):
        return self.PSNR

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.PSNR.assign(0.0)


class StructuralSimilarityIndex(tf.keras.metrics.Metric):
    def __init__(self, name="Structural_similarity_index", **kwargs):
        super(StructuralSimilarityIndex, self).__init__(name=name, **kwargs)
        self.SSIM = self.add_weight(name="SSIM", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        SSIM = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            SSIM = tf.multiply(SSIM, sample_weight)
        self.SSIM.assign(SSIM)

    def result(self):
        return self.SSIM

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.SSIM.assign(0.0)


# if __name__ == '__main__':
