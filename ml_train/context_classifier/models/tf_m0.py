import numpy as np
import tensorflow as tf
from pathlib import Path

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    
    input_layer = tf.reshape(features, [-1, 64, 64, 3])
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )
    
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add softmax_tensor to the graph. It is used for PREDICT
        # and by the logging_hook
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def model_input_fn(input_dir, batch_size):
    
    # extract file_name and label from input paths
    file_names, labels = process_input_dir(input_dir)
    
    # build dataset object
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    # using custom method which shuffles on EACH epoch, check docstring for details
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(len(file_names)))
    dataset = dataset.map(read_image_file, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset


def process_input_dir(input_dir):
    """ 
    Find all image file paths in subdirs, convert to str and extract labels from subdir names
    :param input_dir Path object for parent directory e.g. train
    :returns: list of file paths as str, list of image labels as str
    """
    file_paths = list(input_dir.rglob('*.png'))
    
    file_path_strings = [str(path) for path in file_paths]
    label_strings = [path.parent.name for path in file_paths]
    
    return file_path_strings, label_strings


def read_image_file(file_name, label):
    image_string = tf.read_file(file_name)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)

    # This will convert to float values in [0, 1]
    # Need to check what the "limits" are for this and if we can set them to 0-255
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [64, 64])
    
    return image, label
    



def main():
    # Load training and eval data
    train_dir = Path('ml_train/context_classifier/data/train/')

    # Define estimator
    context_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="ml_train/context_classifier/models/m0"
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    context_classifier.train(
        input_fn=lambda: model_input_fn(train_dir, batch_size=16),
        steps=20,
        hooks=[logging_hook]
    )