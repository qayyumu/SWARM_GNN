import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import glob
import json

class MLP(keras.layers.Layer):
    def __init__(self, units, dropout=0., batch_norm=False, kernel_l2=0., activation='relu', name=None):
        super().__init__(name=name)

        self.hidden_layers = []
        self.dropout_layers = []

        if units:
            for unit in units[:-1]:
                layer = keras.layers.Dense(unit, activation='relu',
                                           kernel_regularizer=keras.regularizers.l2(kernel_l2))
                self.hidden_layers.append(layer)

                dropout_layer = keras.layers.Dropout(dropout)
                self.dropout_layers.append(dropout_layer)

            self.out_layer = keras.layers.Dense(
                units[-1], activation=activation)
        else:
            self.out_layer = keras.layers.Lambda(lambda x: x)

        if batch_norm:
            self.batch_norm = keras.layers.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x, training=False):
        for layer, dropout_layer in zip(self.hidden_layers, self.dropout_layers):
            x = layer(x)
            x = dropout_layer(x, training=training)

        x = self.out_layer(x)
        if self.batch_norm:
            return self.batch_norm(x, training=training)
        return x


class Conv1D(keras.layers.Layer):
    """
    Condense and abstract the time segments.
    """

    def __init__(self, filters, name=None):
        if not filters:
            raise ValueError("'filters' must not be empty")

        super().__init__(name=name)
        # time segment length before being reduced to 1 by Conv1D
        self.seg_len = 2 * len(filters) + 1

        self.conv1d_layers = []
        for i, channels in enumerate(filters):
            layer = keras.layers.TimeDistributed(
                keras.layers.Conv1D(channels, 3, activation='relu', name=name))
            self.conv1d_layers.append(layer)

    def call(self, time_segs):
        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = time_segs
        for conv in self.conv1d_layers:
            encoded_state = conv(encoded_state)

        return encoded_state


class NodePropagator(keras.layers.Layer):
    """
    Pass message between every pair of nodes.
    """

    def call(self, node_states):
        # node_states shape [batch, num_nodes, out_units].
        num_nodes = node_states.shape[1]

        msg_from_source = tf.repeat(tf.expand_dims(
            node_states, 2), num_nodes, axis=2)
        msg_from_target = tf.repeat(tf.expand_dims(
            node_states, 1), num_nodes, axis=1)
        # msg_from_source and msg_from_target in shape [batch, num_nodes, num_nodes, out_units]
        node_msgs = tf.concat([msg_from_source, msg_from_target], axis=-1)

        return node_msgs


class EdgeAggregator(keras.layers.Layer):
    def __init__(self, type='sum', activation=None):
        super().__init__()
        aggregators = {'sum': tf.reduce_sum,
                       'max': tf.reduce_max,
                       'min': tf.reduce_min,
                       'mean': tf.reduce_mean}
        self.aggregator = keras.layers.Lambda(
            lambda x: aggregators[type](x, axis=[1, 3]))

        self.activation = keras.layers.Activation(activation)

    def call(self, edge_msgs, node_states, edges):
        # edge_msg shape [batch, num_nodes, num_nodes, edge_type, out_units]

        # Average messsages of all edge types. Shape becomes [batch, num_nodes, out_units]
        return self.activation(self.aggregator(edge_msgs))


class EdgeEncoder(keras.layers.Layer):
    """
    Propagate messages to edge from the two nodes connected via edge encoders.
    """

    def __init__(self, edge_type, encoder_params):
        super().__init__()

        self.edge_type = edge_type

        self.encoders = [MLP(encoder_params['hidden_units'],
                             encoder_params['dropout'],
                             encoder_params['batch_norm'],
                             encoder_params['kernel_l2'],
                             activation=encoder_params.get('activation'),
                             name=f'edge_encoder_{i}')
                         for i in range(1, self.edge_type+1)]

    def call(self, node_msgs, edges, training=False):
        # `node_msgs` shape [batch, num_nodes*num_nodes, units]
        # `edges` shape [batch, num_nodes, num_nodes, num_edge_label]
        edge_types = tf.expand_dims(edges, axis=-1)
        # Shape [batch, num_nodes, num_nodes, num_edge_label, 1]
        # edge_types = tf.reshape(edges, [-1, num_nodes*num_nodes, num_edge_label, 1])

        encoded_msgs_by_type = []
        for i in range(self.edge_type):
            # mlp_encoder for each edge type.
            encoded_msgs = self.encoders[i](node_msgs, training=training)

            encoded_msgs_by_type.append(encoded_msgs)

        encoded_msgs_by_type = tf.stack(encoded_msgs_by_type, axis=3)
        # Shape [batch, num_nodes, num_nodes, edge_types, units]

        # Only encoded message of the type same as the edge type gets retaind.
        # Force skip 0 type, 0 means no connection, no message.
        edge_msgs = tf.multiply(encoded_msgs_by_type,
                                edge_types[:, :, :, 1:, :])

        return edge_msgs


class NodeDecoder(keras.layers.Layer):
    def __init__(self, params):
        super().__init__()
        self.decoder = MLP(params['hidden_units'],
                           params['dropout'],
                           params['batch_norm'],
                           params['kernel_l2'],
                           name='node_decoder')

    def call(self, node_states, edge_msgs, training=False):
        # node_states and edge_msgs shape [batch, num_nodes, units]
        return self.decoder(tf.concat([node_states, edge_msgs], axis=-1), training=training)


class GraphConv(keras.layers.Layer):
    def __init__(self, graph_size, edge_type, params, name=None):
        super().__init__(name=name)

        self.node_prop = NodePropagator()

        self.edge_aggr = EdgeAggregator(**params['edge_aggr'])

        self.edge_encoder = EdgeEncoder(edge_type, params['edge_encoder'])

        self.node_decoder = NodeDecoder(params['node_decoder'])

    def call(self, node_states, edges, training=False):
        # Propagate node states.
        node_msgs = self.node_prop(node_states)

        # Form edges. Shape [batch, num_edges, edge_type, units]
        edge_msgs = self.edge_encoder(node_msgs, edges, training)

        # Edge aggregation. Shape [batch, num_nodes, units]
        edge_msgs_aggr = self.edge_aggr(edge_msgs, node_states, edges)

        # Update node_states
        node_states = self.node_decoder(node_states, edge_msgs_aggr, training)

        return node_states


class OutLayer(keras.layers.Layer):
    def __init__(self, unit, bound=None, name=None):
        super().__init__(name=name)

        if bound is None:
            self.bound = 1.
            self.dense = keras.layers.Dense(unit)
        else:
            self.bound = np.array(bound, dtype=np.float32)
            self.dense = keras.layers.Dense(unit, 'tanh')

    def call(self, inputs):
        return self.dense(inputs) * self.bound


def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)


def _load_files(file_pattern, dtype, padding=None, pad_dims=None):
    files = sorted(glob.glob(file_pattern))
    print(files)
    if not files:
        raise FileNotFoundError(f"no files matching pattern {file_pattern} found")

    all_data = []
    for f in files:
        data = np.load(f).astype(dtype)
        #print(f)
        if padding is not None and pad_dims is not None:
            pad_shape = [(0, padding - s if i in pad_dims else 0) for i, s in enumerate(data.shape)]
            data = np.pad(data, pad_shape, mode='constant', constant_values=0)

        all_data.append(data)

    return np.concatenate(all_data, axis=0)


def load_data(data_path,no_of_sim, prefix='train',suffix='',more_sim=False, size=None, padding=None, load_time=False):
    if not os.path.exists(data_path):
        raise ValueError(f"path '{data_path}' does not exist")
    # Load edge data.
    edge_file_pattern = os.path.join(data_path, f'{prefix}_edge*.npy')
    all_edges = _load_files(edge_file_pattern, np.int, padding, pad_dims=(1, 2))

    if (not more_sim):
        newsuffix = suffix+str(0)
        # Load timeseries data.
        timeseries_file_pattern = os.path.join(data_path, f'{prefix}_timeseries{newsuffix}.npy')
        all_data = _load_files(timeseries_file_pattern, np.float32, padding=padding, pad_dims=(2,))

        shuffled_idx = np.random.permutation(len(all_data))
        # Truncate data samples if `size` is given.
        if size:
            samples = shuffled_idx[:size]

            all_data = all_data[samples]
            all_edges = all_edges[samples]

        # Load time labels only when required.
        if load_time:
            time_file_pattern = os.path.join(data_path, f'{prefix}_time*.npy')
            all_times = _load_files(time_file_pattern, np.float32)

            if size:
                samples = shuffled_idx[:size]

                all_times = all_times[samples]

            return all_data, all_edges, all_times

        return all_data, all_edges
    
    else:
        all_data=[]
        all_times=[]
        for i in range(no_of_sim):
            newsuffix = suffix+str(i)
            # Load timeseries data.
            timeseries_file_pattern = os.path.join(data_path, f'{prefix}_timeseries{newsuffix}.npy')
            data = _load_files(timeseries_file_pattern, np.float32, padding=padding, pad_dims=(2,))
            all_data.append(data)

            shuffled_idx = np.random.permutation(len(all_data))
            # Truncate data samples if `size` is given.
            if size:
                samples = shuffled_idx[:size]

                all_data = all_data[samples]
                all_edges = all_edges[samples]

            # Load time labels only when required.
            if load_time:
                time_file_pattern = os.path.join(data_path, f'{prefix}_time{newsuffix}*.npy')
                times = _load_files(time_file_pattern, np.float32)
                all_times.append(times)

                if size:
                    samples = shuffled_idx[:size]

                    all_times = all_times[samples]

                return all_data, all_edges, all_times

        return all_data, all_edges



def preprocess_data(data, seg_len=1, pred_steps=1, edge_type=1, ground_truth=True):
    time_series, edges = data[:2]
    print(time_series.shape)
    time_steps, num_nodes, ndims = time_series.shape[1:]
    
    if (seg_len + pred_steps > time_steps):
        if ground_truth:
            raise ValueError('time_steps in data not long enough for seg_len and pred_steps')
        else:
            stop = 1
    else:
        stop = -pred_steps

    edge_label = edge_type + 1  # Accounting for "no connection"

    # time_series shape [num_sims, time_steps, num_nodes, ndims]
    # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_nodes, ndims]
    time_segs_stack = stack_time_series(time_series[:, :stop, :, :],
                                        seg_len)
    time_segs = time_segs_stack.reshape([-1, seg_len, num_nodes, ndims])
    if ground_truth:
        # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_nodes, ndims]
        expected_time_segs_stack = stack_time_series(time_series[:, seg_len:, :, :],
                                                     pred_steps)
        assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
                == time_steps - seg_len - pred_steps + 1)
        expected_time_segs = expected_time_segs_stack.reshape([-1, pred_steps, num_nodes, ndims])
    else:
        expected_time_segs = None

    edges_one_hot = one_hot(edges, edge_label, np.float32)
    edges_one_hot = np.repeat(edges_one_hot, time_segs_stack.shape[1], axis=0)

    if len(data) > 2:
        time_stamps = data[2]

        time_stamps_stack = stack_time_series(time_stamps[:, :stop], seg_len)
        time_stamps_segs = time_stamps_stack.reshape([-1, seg_len])

        if ground_truth:
            expected_time_stamps_stack = stack_time_series(
                time_stamps[:, seg_len:], pred_steps)
            expected_time_stamps_segs = expected_time_stamps_stack.reshape([-1, pred_steps])
        else:
            expected_time_stamps_segs = None

        return [time_segs, edges_one_hot], expected_time_segs, [time_stamps_segs, expected_time_stamps_segs]

    return [time_segs, edges_one_hot], expected_time_segs


def off_diag_matrix(n):
    return np.ones((n, n)) - np.eye(n)


def one_hot(labels, num_classes, dtype=np.int):
    identity = np.eye(num_classes, dtype=dtype)
    one_hots = identity[labels.reshape(-1)]
    return one_hots.reshape(labels.shape + (num_classes,))


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(checkpoint):
        model.load_weights(checkpoint)


def save_model(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, 'weights.h5')

    model.save_weights(checkpoint)

    return tf.keras.callbacks.ModelCheckpoint(checkpoint, save_weights_only=True)


def load_model_params(config):
    with open(config) as f:
        model_params = json.load(f)

    seg_len = 2 * len(model_params['cnn']['filters']) + 1
    model_params['time_seg_len'] = seg_len
    model_params.setdefault('edge_type', 1)
    model_params.setdefault('output_bound')
    model_params.setdefault('edge_aggr', {})

    return model_params

