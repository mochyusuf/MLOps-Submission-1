import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Sentiment"
FEATURE_KEY = "OriginalTweet"


def transformed_name(key):
    return f"{key}_xf"


def preprocessing_fn(inputs):
    outputs = {}
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    return outputs