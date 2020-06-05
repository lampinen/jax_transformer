import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random, ops
from jax.experimental import optimizers
from jax.nn import softmax, one_hot

from datasets import build_reversing_dataset


def leaky_relu(x, neg_slope=0.2):
    return jnp.where(x >= 0, x, neg_slope * x)


def layer_norm(x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    sd = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / sd


def positional_encodings(length, dimensionality, constant_fac=1000):
    positions = jnp.arange(length, dtype=jnp.float32)
    constants = (2./dimensionality) * (jnp.arange(dimensionality, dtype=jnp.float32) // 2.)
    constants = 1./jnp.exp(constants * jnp.log(constant_fac)) 
    offsets = jnp.zeros_like(constants) 
    offsets = ops.index_update(offsets, ops.index[::2], jnp.pi / 2.)  # sin -> cos 
    encodings = jnp.outer(positions, constants)
    encodings += offsets
    encodings = jnp.sin(encodings)
    return encodings


def single_attention_head(params, inputs, Q_inputs=None, mask=None): 
    """params = tuple of weight matrices for Q, K, V,
    
    Q_inputs: if not none, queries will be constructed from these.
    mask: if not None, should be binary mask for attention selections."""
    qw, kw, vw = params
    
    if Q_inputs is not None:
        Q = jnp.dot(Q_inputs, qw)
    else:
        Q = jnp.dot(inputs, qw)
    K = jnp.dot(inputs, kw)
    V = jnp.dot(inputs, vw)

    scale = jnp.sqrt(inputs.shape[-1])

    attention_selections = jnp.matmul(Q, jnp.transpose(K, axes=[0,2,1])) / scale

    if mask is not None:  # mask out "off" locations with -inf
        attention_selections = jnp.where(mask, attention_selections, -jnp.inf)

    attention_selections = softmax(attention_selections, axis=-1) 

    outputs = jnp.matmul(attention_selections, V)

    return outputs


def two_layer_feed_forward(params, inputs):
    w1, b1, w2, b2 = params
    hidden = leaky_relu(jnp.dot(inputs, w1) + b1)
    outputs = jnp.dot(hidden, w2) + b2
    return outputs


def multi_head_attention(att_params, out_params, inputs, Q_inputs=None, mask=None):
    """att_params should be list of single-head tuples, out_params
    
    causal_mask: prevent from attending right of current point
    """

    results = [] 
    for param_set in att_params:
        results.append(single_attention_head(param_set, inputs,
                                             Q_inputs=Q_inputs, mask=mask))

    results = jnp.concatenate(results, axis=-1)

    ow, ob = out_params
    results = jnp.dot(results, ow) + ob 
    return results
    

def encoder_layer(params, inputs): 
    attention_params, ff_params = params

    att_params, out_params = attention_params
    hidden_1 = multi_head_attention(att_params, out_params, inputs) + inputs
    hidden_1 = layer_norm(hidden_1)

    output = two_layer_feed_forward(ff_params, hidden_1) + hidden_1 
    output = layer_norm(output)

    return output
#
#
#def get_attention_masks(sequence_length):
#    mask = jnp.zeros([sequence_length, sequence_length],
#                     dtype=jnp.bool)
#    for i in range(sequence_length):
#        mask[i, :i] = True
#    # add batchdim 
#    mask = jnp.expand_dims(mask, axis=0)


def decoder_layer(params, inputs, encoder_outputs):
    """inputs are previous outputs"""
    attention_1_params, attention_2_params, ff_params = params

    att_1_params, out_1_params = attention_1_params

    hidden_1 = multi_head_attention(att_1_params, out_1_params, inputs) + inputs
    hidden_1 = layer_norm(hidden_1)

    att_2_params, out_2_params = attention_2_params

    hidden_2 = multi_head_attention(att_2_params, out_2_params, inputs=encoder_outputs, Q_inputs=hidden_1) + hidden_1 
    hidden_2 = layer_norm(hidden_2)

    output = two_layer_feed_forward(ff_params, hidden_2) + hidden_2 
    output = layer_norm(output)

    return output
      

def transformer(params, inputs, out_seq_length, num_symbols):
    one_hot_inputs = one_hot(inputs, num_classes=num_symbols)

    one_hot_outputs = jnp.zeros([inputs.shape[0], out_seq_length + 1, num_symbols + 1])
    #one_hot_outputs[:, 0, -1] = 1.  # start symbol
    one_hot_outputs = ops.index_update(one_hot_outputs, ops.index[:, 0, -1], 1.)
    output_logits = jnp.zeros([inputs.shape[0], out_seq_length, num_symbols])

    (input_embeddings, output_embeddings,
     encoder_params_list, decoder_params_list,
     output_params) = params
    output_w, output_b = output_params

    encoder_results = jnp.dot(one_hot_inputs, input_embeddings)
    encoder_results += positional_encodings(encoder_results.shape[-2], encoder_results.shape[-1]) 
    for encoder_layer_i, this_enc_params in enumerate(encoder_params_list):
        encoder_results = encoder_layer(this_enc_params, encoder_results)

    for i in range(out_seq_length):
        decoder_results = jnp.dot(one_hot_outputs[:, :i + 1, :], output_embeddings) 
        decoder_results += positional_encodings(decoder_results.shape[-2], decoder_results.shape[-1]) 
        for decoder_layer_i, this_dec_params in enumerate(decoder_params_list):
            decoder_results = decoder_layer(this_dec_params, 
                                            decoder_results,
                                            encoder_results)

        this_step_results = jnp.dot(decoder_results, output_w) + output_b
        this_step_results = this_step_results[:, -1, :]  # previous outputs already produced
        
        output_logits = ops.index_update(output_logits, ops.index[:, i, :], this_step_results) 
        one_hot_outputs = ops.index_update(
            one_hot_outputs,
            ops.index[:, i + 1, :],
            one_hot(jnp.argmax(this_step_results, axis=-1),
                    num_classes=num_symbols + 1))

    one_hot_outputs = one_hot_outputs[:, 1:, :-1]  # chop start symbol
    return output_logits, one_hot_outputs


def init_transformer_params(init_scale, num_encoder_layers, num_decoder_layers,
                            dimensionality, num_heads, head_dimensionality, 
                            num_symbols, rng=npr.RandomState(0)):

    input_embeddings = init_scale * rng.randn(num_symbols, dimensionality)
    output_embeddings = init_scale * rng.randn(num_symbols + 1, dimensionality)
    encoder_params_list = []
    for i in range(num_encoder_layers):
        att_params = []
        for j in range(num_heads):
            qw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            kw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            vw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            this_att_params = (qw, kw, vw)
            att_params.append(this_att_params)
        
        ow = init_scale * rng.randn(num_heads * head_dimensionality, dimensionality)  
        ob = init_scale * rng.randn(dimensionality)
        out_params = (ow, ob) 

        w1 = init_scale * rng.randn(dimensionality, dimensionality)  
        b1 = init_scale * rng.randn(dimensionality)
        w2 = init_scale * rng.randn(dimensionality, dimensionality)  
        b2 = init_scale * rng.randn(dimensionality)
        ff_params = (w1, b1, w2, b2)

        attention_params = (att_params, out_params)
        this_layer_params = (attention_params, ff_params)
        encoder_params_list.append(this_layer_params)

    decoder_params_list = []
    for i in range(num_decoder_layers):
        att_1_params = []
        for j in range(num_heads):
            qw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            kw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            vw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            this_att_params = (qw, kw, vw)
            att_1_params.append(this_att_params)

        ow = init_scale * rng.randn(num_heads * head_dimensionality, dimensionality)  
        ob = init_scale * rng.randn(dimensionality)
        out_1_params = (ow, ob) 

        att_2_params = []
        for j in range(num_heads):
            qw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            kw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            vw = init_scale * rng.randn(dimensionality, head_dimensionality)  
            this_att_params = (qw, kw, vw)
            att_2_params.append(this_att_params)

        ow = init_scale * rng.randn(num_heads * head_dimensionality, dimensionality)  
        ob = init_scale * rng.randn(dimensionality)
        out_2_params = (ow, ob) 

        w1 = init_scale * rng.randn(dimensionality, dimensionality)  
        b1 = init_scale * rng.randn(dimensionality)
        w2 = init_scale * rng.randn(dimensionality, dimensionality)  
        b2 = init_scale * rng.randn(dimensionality)
        ff_params = (w1, b1, w2, b2)

        this_layer_params = ((att_1_params, out_1_params), 
                             (att_2_params, out_2_params), 
                             ff_params)
        decoder_params_list.append(this_layer_params)

    
    fow = init_scale * rng.randn(dimensionality, num_symbols)  
    fob = init_scale * rng.randn(num_symbols)
    final_output_params = (fow, fob) 

    params = [input_embeddings, output_embeddings, encoder_params_list,
              decoder_params_list, final_output_params]
    return params 



if __name__ == "__main__":
    #print(positional_encodings(10, 6, 20))
    seq_length = 5
    num_encoder_layers = 3
    num_decoder_layers = 2
    dimensionality = 64
    num_heads = 2
    head_dimensionality = 32
    num_symbols = 10
    init_scale = 0.05
    batch_size = 10
    num_train = 1000
    num_test = 100
    learning_rate = 1e-4
    num_epochs = 100000

#    # larger model, longer sequence test
#    seq_length = 10
#    num_encoder_layers = 6
#    num_decoder_layers = 6
#    dimensionality = 256
#    num_heads = 4
#    head_dimensionality = 64

    init_params = init_transformer_params(
        init_scale, num_encoder_layers, num_decoder_layers, dimensionality,
        num_heads, head_dimensionality, num_symbols) 

    # tweaked from an example
    def loss(params, batch):
        inputs, targets = batch
        one_hot_targets = one_hot(targets, num_classes=num_symbols) 
        preds, _ = transformer(params, inputs, out_seq_length=seq_length, num_symbols=num_symbols)
        return -jnp.mean(jnp.sum(jnp.sum(preds * one_hot_targets, axis=-1), axis=-1))

    def get_predictions(params, inputs):
        logits, one_hot_preds = transformer(params, inputs, out_seq_length=seq_length, num_symbols=num_symbols)
        return jnp.argmax(one_hot_preds, axis=-1)

    def get_accuracy(params, inputs, targets):
        predicted_class = get_predictions(params, inputs)
        return jnp.mean(predicted_class == targets)
    
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)

    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    dataset = build_reversing_dataset(num_train=num_train, num_test=num_test,
                                      num_ints=num_symbols, seq_length=seq_length)

    params = get_params(opt_state)
    print("Epoch: {}, Train (subset) loss: {}, Train (subset) accuracy: {}, test_accuracy: {}".format(
        0, 
        loss(params, (dataset["train"]["inputs"][-num_test:, :], dataset["train"]["outputs"][-num_test:, :])),
        get_accuracy(params, dataset["train"]["inputs"][-num_test:, :], dataset["train"]["outputs"][-num_test:, :]),
        get_accuracy(params, dataset["test"]["inputs"], dataset["test"]["outputs"])))
    print(dataset["test"]["inputs"][:5, :])
    print(get_predictions(params, dataset["test"]["inputs"][:5, :]))

    batch_count = 0
    for epoch_i in range(1, num_epochs):
        for batch_i in range(num_train // batch_size):
            opt_state = update(batch_count, opt_state,
                (dataset["train"]["inputs"][batch_i * batch_size: (batch_i + 1) * batch_size:, :], 
                 dataset["train"]["outputs"][batch_i * batch_size: (batch_i + 1) * batch_size:, :]))
            batch_count += 1

        params = get_params(opt_state)
        print("Epoch: {}, Train (subset) loss: {}, Train (subset) accuracy: {}, test_accuracy: {}".format(
            epoch_i, 
            loss(params, (dataset["train"]["inputs"][-num_test:, :], dataset["train"]["outputs"][-num_test:, :])),
            get_accuracy(params, dataset["train"]["inputs"][-num_test:, :], dataset["train"]["outputs"][-num_test:, :]),
            get_accuracy(params, dataset["test"]["inputs"], dataset["test"]["outputs"])))
        print(dataset["test"]["inputs"][:5, :])
        print(get_predictions(params, dataset["test"]["inputs"][:5, :]))
