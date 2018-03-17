import numpy as np 
import tensorflow as tf 

# straight from the TF example, modified to work with 12 classes istead of 10
def cnn_model_fn(features, labels, mode, params):
    if params['environment'] == 'TouchWandLessRewardEnv' or params['environment'] == 'TouchWandEnv':
        input_layer = tf.reshape(features["x"], [-1, 100, 100, 1])
    elif params['environment'] == 'FingerJointEnv':
        input_layer = tf.reshape(features["x"], [-1, 300, 300, 1])
    else:
        raise ValueError('Unknown Environment. Cannot Reshape Inputs with Unkonwn Image Dimensions')
        
    num_classes = params['num_classes']
    conv1 = tf.layers.conv2d(inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    if params['environment'] == 'TouchWandLessRewardEnv' or params['environment'] == 'TouchWandEnv':
        pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64])
    elif params['environment'] == 'FingerJointEnv':
        pool2_flat = tf.reshape(pool2, [-1, 75 * 75 * 64])
    else:
        raise ValueError('Cannot Flatten Pooling Layer with Unknown Image Dimensions')

    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4,
                    training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=num_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def cnn_lstm_model_fn(features, labels, mode, params):
    if params['environment'] == 'TouchWandLessRewardEnv' or params['environment'] == 'TouchWandEnv':
        input_layer = tf.reshape(features["x"], [params['batch_size'], 10000, 1])
    else:
        raise ValueError('Unknown Environment. Cannot Reshape Inputs with Unkonwn Image Dimensions')
        
    num_classes = params['num_classes']
    conv1 = tf.layers.conv1d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5],
                             padding='same',
                             activation=None)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(6)
    output, state = tf.nn.dynamic_rnn(lstm_cell, conv1, dtype=tf.float32)
    output = tf.reduce_sum(output, axis=1)

    logits = tf.layers.dense(inputs=output, units=num_classes)

    predictions = { "classes": tf.argmax(logits, axis=1) }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            labels=labels, logits=logits))

    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=0.001,
        optimizer='Adam')  

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={'logits': logits, 'predictions': predictions['classes']},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={'accuracy': tf.metrics.accuracy(labels, predictions['classes'])})