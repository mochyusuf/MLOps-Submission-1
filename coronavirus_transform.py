import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Sentiment"
LABEL_DIMENSION = 5
FEATURE_KEY = "OriginalTweet"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(LABEL_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    label_indices = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])
    label_one_hot = tf.one_hot(label_indices, depth=LABEL_DIMENSION)
    outputs[transformed_name(key=LABEL_KEY)] = tf.cast(
        tf.reshape(label_one_hot, [-1, LABEL_DIMENSION]),
        tf.float32,
    )
    return outputs