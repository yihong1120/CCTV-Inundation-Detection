import os
import json
import random

import tensorflow as tf
import matplotlib.pyplot as plt


def get_class_indices(root: str):
    """
    Generate class indices for the dataset classes.
    :param root: The root directory of the dataset.
    :return: A dictionary mapping class names to indices.
    """
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    return class_indices


def save_class_indices(class_indices: dict, file_path: str = 'class_indices.json'):
    """
    Save class indices to a JSON file.
    :param class_indices: A dictionary of class indices.
    :param file_path: The file path to save the JSON data.
    """
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(file_path, 'w') as json_file:
        json_file.write(json_str)


def split_data(root: str, class_indices: dict, val_rate: float = 0.2):
    """
    Split the dataset into training and validation sets.
    :param root: The root directory of the dataset.
    :param class_indices: A dictionary of class indices.
    :param val_rate: The proportion of validation data.
    :return: Paths and labels for training and validation images.
    """
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]

    for cla in class_indices.keys():
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    return train_images_path, train_images_label, val_images_path, val_images_label, every_class_num


def plot_class_distribution(flower_class: list, every_class_num: list):
    """
    Plot the distribution of classes in the dataset.
    :param flower_class: A list of class names.
    :param every_class_num: A list of the number of images per class.
    """
    plt.bar(range(len(flower_class)), every_class_num, align='center')
    plt.xticks(range(len(flower_class)), flower_class)
    for i, v in enumerate(every_class_num):
        plt.text(x=i, y=v + 5, s=str(v), ha='center')
    plt.xlabel('image class')
    plt.ylabel('number of images')
    plt.title('flower class distribution')
    plt.show()


def process_image(img_path, label, im_height, im_width, augment=False):
    """
    Process an image file into a format suitable for training/validation.
    :param img_path: The file path of the image.
    :param label: The label of the image.
    :param im_height: The height to resize the image to.
    :param im_width: The width to resize the image to.
    :param augment: Whether to apply data augmentation.
    :return: The processed image and its label.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, im_height, im_width)
    if augment:
        image = tf.image.random_flip_left_right(image)
    return image, label


def configure_for_performance(ds, shuffle_size, batch_size, shuffle=False):
    """
    Configure the dataset for performance.
    :param ds: The dataset to configure.
    :param shuffle_size: The buffer size for shuffling.
    :param batch_size: The batch size for batching the dataset.
    :param shuffle: Whether to shuffle the dataset.
    :return: The configured dataset.
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def generate_ds(data_root: str, im_height: int, im_width: int, batch_size: int, val_rate: float = 0.1):
    """
    Generate training and validation datasets.
    :param data_root: The root directory of the dataset.
    :param im_height: The height of the images.
    :param im_width: The width of the images.
    :param batch_size: The batch size.
    :param val_rate: The proportion of validation data.
    :return: The training and validation datasets.
    """
    class_indices = get_class_indices(data_root)
    save_class_indices(class_indices)
    train_img_path, train_img_label, val_img_path, val_img_label, every_class_num = split_data(data_root, class_indices, val_rate)

    plot_class_distribution(class_indices.keys(), every_class_num)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_tensor_slices((train_img_path, train_img_label))
    train_ds = train_ds.map(lambda x, y: process_image(x, y, im_height, im_width, augment=True), num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, len(train_img_path), batch_size, shuffle=True)

    val_ds = tf.data.Dataset.from_tensor_slices((val_img_path, val_img_label))
    val_ds = val_ds.map(lambda x, y: process_image(x, y, im_height, im_width), num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, len(val_img_path), batch_size)

    return train_ds, val_ds