import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import build_reversing_dataset

def get_word_embeddings(inputs, vocab_size, sizeality, reuse=True):
    with tf.variable_scope("word_embeddings", reuse=reuse):
        word_embeddings = tf.get_variable(
            "embeddings", shape=[vocab_size, sizeality])

    embedded_language = tf.nn.embedding_lookup(word_embeddings,
                                               inputs)

    return embedded_language


def layer_norm(x, scope="layer_norm", reuse=True):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x_norm = x - mean
    sd = tf.sqrt(tf.reduce_mean(tf.square(x_norm), axis=-1, keepdims=True))
    with tf.variable_scope(scope, reuse=reuse):
        gain = tf.get_variable("gain", shape=[x.get_shape()[-1]])
        bias = tf.get_variable("bias", shape=[x.get_shape()[-1]])
    return (gain * x_norm / sd) + bias


def positional_encodings(length, size, constant_fac=1000):
    positions = np.arange(length, dtype=np.float32)
    constants = (2./size) * (np.arange(size, dtype=np.float32) // 2.)
    constants = 1./np.exp(constants * np.log(constant_fac))
    offsets = np.zeros_like(constants)
    offsets[::2] += np.pi / 2.  # sin -> cos
    encodings = np.outer(positions, constants)
    encodings += offsets
    encodings = np.sin(encodings)
    return encodings


def single_attention_head(inputs, scope, Q_inputs=None, head_size=128, reuse=True):
    with tf.variable_scope(scope, reuse=reuse):
        if Q_inputs is None:
            Q = slim.fully_connected(
                inputs, head_size,
                activation_fn=None)
        else:  # separate query inputs 
            Q = slim.fully_connected(
                Q_inputs, head_size,
                activation_fn=None)

        K = slim.fully_connected(
            inputs, head_size,
            activation_fn=None)
        V = slim.fully_connected(
            inputs, head_size,
            activation_fn=None)

        scale = 1. / np.sqrt(head_size)

        attention_selections = scale * tf.matmul(Q, K, transpose_b=True) 

        attention_selections = tf.nn.softmax(attention_selections, axis=-1)

        outputs = tf.matmul(attention_selections, V)

        return outputs, attention_selections, Q, K, V


def two_layer_feed_forward(inputs, scope, size=512, reuse=True): 
    with tf.variable_scope(scope, reuse=reuse):
        hidden = slim.fully_connected(inputs, size,
                                      activation_fn=tf.nn.leaky_relu)
        output = slim.fully_connected(hidden, size,
                                      activation_fn=None)
    return output


def multi_head_attention(inputs, scope, Q_inputs=None, num_heads=4,
                         head_size=128, shared_size=512,
                         reuse=True):
    """att_params should be list of single-head tuples, out_params
    
    causal_mask: prevent from attending right of current point
    """

    results = []
    with tf.variable_scope(scope, reuse=reuse):
        for head_i in range(num_heads):
            results.append(single_attention_head(inputs, Q_inputs=Q_inputs,
                                                 scope="head_%i" % head_i, 
                                                 head_size=head_size,
                                                 reuse=reuse)[0])

        result_values = tf.concat([x[0] for x in results], axis=-1)

        result_values = slim.fully_connected(result_values, shared_size,
                                             activation_fn=None) 
    return result_values, results


def encoder_layer(inputs, scope, num_heads=4, head_size=128, shared_size=512,
                  reuse=True):
    with tf.variable_scope(scope, reuse=reuse):
        hidden, _ = multi_head_attention(
            inputs, scope="attention", num_heads=num_heads, head_size=head_size,
            shared_size=shared_size, reuse=reuse)

        hidden = layer_norm(hidden + inputs, scope="layer_norm_1", reuse=reuse)

        output = two_layer_feed_forward(hidden, scope="fc", size=shared_size,
                                        reuse=reuse)

        output = layer_norm(output + hidden, scope="layer_norm_2", reuse=reuse)

    return output


def decoder_layer(inputs, encoder_outputs, scope, num_heads=4, head_size=128,
                  shared_size=512, reuse=True):
    with tf.variable_scope(scope, reuse=reuse):
        hidden_1, _ = multi_head_attention(
            inputs, scope="attention_1", num_heads=num_heads, head_size=head_size,
            shared_size=shared_size, reuse=reuse)

        hidden_1 = layer_norm(hidden_1 + inputs, scope="layer_norm_1", reuse=reuse)

        hidden_2, _ = multi_head_attention(
            encoder_outputs, Q_inputs=hidden_1, scope="attention_2", num_heads=num_heads, head_size=head_size,
            shared_size=shared_size, reuse=reuse)

        hidden_2 = layer_norm(hidden_2 + hidden_1, scope="layer_norm_2", reuse=reuse)

        output = two_layer_feed_forward(hidden_2, scope="fc", size=shared_size,
                                        reuse=reuse)

        output = layer_norm(output + hidden_2, scope="layer_norm_3", reuse=reuse)

    return output


def masked_xe_loss(logits, targets, mask):
    unmasked_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
    masked_loss = tf.where(mask, unmasked_loss,
                           tf.zeros_like(unmasked_loss))

    return masked_loss


class seq2seq_transformer(object):
    def __init__(self, input_seq_length, output_seq_length, num_symbols,
                 num_encoder_layers=3, num_decoder_layers=3, num_heads=4,
                 head_size=128, shared_size=512, batch_size=10):
        self.batch_size = batch_size
        self.input_ph = tf.placeholder(
            tf.int32, shape=[None, input_seq_length])
        self.target_ph = tf.placeholder(
            tf.int32, shape=[None, output_seq_length])
        self.target_mask_ph = tf.placeholder(
            tf.bool, shape=[None, output_seq_length])

        self.lr_ph = tf.placeholder(tf.float32)

        # word embeddings
        self.embedded_inputs = get_word_embeddings(
            self.input_ph, num_symbols, shared_size, reuse=False)

        # one-hot targets
        oh_target = tf.one_hot(self.target_ph,
                               depth=num_symbols)

        self.embedded_positional_inputs = self.embedded_inputs + positional_encodings(input_seq_length, shared_size) 
        
        encoder_results = self.embedded_positional_inputs
        with tf.variable_scope("encoder", reuse=False):
            for encoder_layer_i in range(num_encoder_layers):
                encoder_results = encoder_layer(
                    encoder_results, scope="encoder_layer_%i" % encoder_layer_i,
                    num_heads=num_heads, head_size=head_size,
                    shared_size=shared_size, reuse=False)
        
        
        batch_size = tf.shape(self.input_ph)[0]
        with tf.variable_scope("decoder", reuse=False):
            start_symbol = tf.get_variable("start_symbol", [1, 1, shared_size]) 
        decoder_input_embeddings = tf.tile(start_symbol, [batch_size, 1, 1])
        these_position_encodings = positional_encodings(output_seq_length,
                                                        shared_size)

        output_logits = []
        for output_i in range(output_seq_length): 
            with tf.variable_scope("decoder", reuse=output_i > 0):
                decoder_results = decoder_input_embeddings 
                decoder_results += these_position_encodings[:output_i + 1, :] 
                for decoder_layer_i in range(num_decoder_layers):
                    decoder_results = decoder_layer(
                        decoder_results, encoder_results,
                        scope="decoder_layer_%i" % decoder_layer_i,
                        num_heads=num_heads, head_size=head_size,
                        shared_size=shared_size, reuse=output_i > 0)

                decoder_input_embeddings = tf.concat(
                    [decoder_input_embeddings, decoder_results[:, -1:, :]], axis=1)

                this_output_logits = slim.fully_connected(
                    decoder_results[:, -1, :], num_symbols,
                    activation_fn=None)
                output_logits.append(this_output_logits)

        self.output_logits = tf.stack(output_logits, axis=1)

        self.loss = masked_xe_loss(logits=self.output_logits,
                                   targets=oh_target,
                                   mask=self.target_mask_ph)

        self.total_loss = tf.reduce_mean(self.loss)

        def get_accuracy(outputs, targets, mask):
            """targets should be int"""
            hard_outputs = tf.argmax(outputs, axis=-1, output_type=tf.int32)
            outputs_equal_target = tf.equal(hard_outputs, targets)
            return tf.reduce_mean(
                tf.cast(tf.boolean_mask(outputs_equal_target,
                                        mask),
                        tf.float32))

        self.total_accuracy = get_accuracy(
            outputs=self.output_logits,
            targets=self.target_ph,
            mask=self.target_mask_ph)

        self.optimizer = tf.train.AdamOptimizer(self.lr_ph)

        self.train_op = self.optimizer.minimize(self.total_loss)

        self._sess_and_init()

    def _sess_and_init(self):
        # Saver
        self.saver = tf.train.Saver()

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

    def save_parameters(self, filename):
        self.saver.save(self.sess, filename)

    def restore_parameters(self, filename):
        self.saver.restore(self.sess, filename)

    def build_feed_dict(self, inputs, targets=None, masks=None, lr=None):     
        feed_dict = {}
        feed_dict[self.input_ph] = inputs
        feed_dict[self.target_ph] = targets
        feed_dict[self.target_mask_ph] = masks
        feed_dict[self.lr_ph] = lr
        return feed_dict

    def do_eval(self, dataset):
        batch_size = self.batch_size
        results = {}
        for subset_name, subset in dataset.items():
            results[subset_name] = {}
            loss = 0.
            accuracy = 0.
            num_batches = int(np.ceil(len(subset["inputs"]) / float(batch_size)))
            for batch_i in range(num_batches): 
                feed_dict = self.build_feed_dict(
                    inputs=subset["inputs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    targets=subset["outputs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    masks=subset["masks"][batch_i * batch_size:(batch_i + 1) * batch_size])
                this_loss, this_accuracy = self.sess.run(
                    [self.total_loss, self.total_accuracy],
                    feed_dict=feed_dict)
                
                loss += this_loss
                accuracy += this_accuracy
            loss /= num_batches
            accuracy /= num_batches
            results[subset_name]["loss"] = loss
            results[subset_name]["accuracy"] = accuracy
        return results


    def do_training(self, dataset, num_epochs, lr):
        print("Epoch 0:")
        print(self.do_eval(dataset))

        train_data = dataset["train"]
        batch_size = self.batch_size
        num_batches = int(np.ceil(len(train_data["inputs"]) / float(batch_size)))
        for epoch_i in range(1, num_epochs + 1):
            for batch_i in range(num_batches):
                feed_dict = self.build_feed_dict(
                    inputs=train_data["inputs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    targets=train_data["outputs"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    masks=train_data["masks"][batch_i * batch_size:(batch_i + 1) * batch_size],
                    lr=lr)
                self.sess.run(self.train_op, feed_dict=feed_dict)
            print("Epoch %i:" % epoch_i)
            print(self.do_eval(dataset))



if __name__ == "__main__":

#    x = tf.random.uniform([10, 5, 512])
#    y = tf.random.uniform([10, 7, 512])
#    print(layer_norm(x, reuse=False))
#    print(single_attention_head(x, "head_0", reuse=False))
#
#    print(multi_head_attention(x, "blah", reuse=False))
#    print(multi_head_attention(x, "blah2", Q_inputs=y, reuse=False))
#    print(encoder_layer(x, scope="encoder_layer_1", reuse=False))
#    print(decoder_layer(x, y, scope="decoder_layer_1", reuse=False))
    dataset = build_reversing_dataset(
        num_train=1000, num_test=100, num_ints=10, seq_length=5) 
    for subset in dataset:
        dataset[subset]["masks"] = np.ones(dataset[subset]["outputs"].shape, dtype=np.bool)

    model = seq2seq_transformer(
        input_seq_length=5, output_seq_length=5, num_symbols=10,
        num_encoder_layers=3, num_decoder_layers=3, num_heads=4,
        head_size=64, shared_size=128, batch_size=10) 

    model.do_training(dataset, num_epochs=10000, lr=1e-4)
