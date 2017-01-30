import tensorflow as tf


class TextDCNN(object):
    """
    A CNN for NLP tasks. Architecture is as follows:
    Embedding layer, conv layer, max-pooling and softmax layer
    """

    def __init__(self, sequence_lengths, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        """
        Makes a new CNNClassifier
        Args:
            sequence_length: The length of each sentence
            num_classes: Number of classes in the output layer (positive and negative would be 2 classes)
            vocab_size: The size of the vocabulary, needed to define the size of the embedding layer
            embedding_size: Dimensionality of the embeddings
            filter_sizes: Number of words the convolutional filters will cover, there will be num_filters for each size
            specified.
            num_filters: The number of filters per filter size.

        Returns: A new TextDCNN with the given parameters.

        """
        # Define the inputs and the dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Runs the operations on the CPU and organizes them into an embedding scope
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            W = tf.Variable(  # Make a 4D tensor to store batch, width, height, and channel
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W"
            )

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Conv layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # W is the filter matrix
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Max-pooling layer over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )

                print("SL:")
                print(sequence_lengths[i])
                print("H:")
                print(h)
                print("K:")
                print(sequence_lengths[i] - filter_size + 1)

                # pooled = tf.reshape(pooled, [-1, 1, 1, num_filters])

                print("POOLED:")
                print(pooled)
                print()

                pooled_outputs.append(pooled)

        # Combine all of the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        # pooled_outputs = [tf.pad(p, [[0, int(max_shape.get_shape()[0]) - int(p.get_shape()[0])], [0, 0], [0, 0], [0, 0]]) for p in pooled_outputs]
        # pooled_outputs = [tf.reshape(p, [-1, 1, 1, num_filters]) for p in pooled_outputs]

        # pooled_outputs = [tf.reshape(out, [-1, 1, 1, self.max_length]) for out in pooled_outputs]

        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print("here")
        print(self.h_pool_flat)
        # self.h_pool_flat = tf.reshape(self.h_pool, [max(sequence_lengths), num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            # casted = tf.cast(self.dropout_keep_prob, tf.int32)
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop = tf.reshape(self.h_drop, [-1, num_filters_total])

        # Do raw predictions (no softmax)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # xw_plus_b(...) is just Wx + b matmul alias
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # softmax_cross_entropy_with_logits(...) calculates cross-entropy loss
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            '''print("here")
            print(losses.get_shape())
            print(self.scores.get_shape())
            print(self.input_y.get_shape())'''
            self.loss = tf.reduce_mean(losses)

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
