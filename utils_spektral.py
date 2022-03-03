import numpy as np
import os
from spektral.data import Dataset, Graph
import tensorflow as tf
import glob
from tensorflow import keras
import json

## MyDataSet not used anywhere
class MyDataset(Dataset):
    """
    Turn swarm into a graph
    """
    def __init__(self, nodes, feats, **kwargs):
        self.nodes = nodes
        self.feats = feats

        super().__init__(**kwargs)

    def download(self,ARGS,prefix,X):
        # Create the directory
        os.mkdir(self.path)
        # Write the data to file
        x = X
        a= load_edge_data(ARGS.data_dir,
                                   prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding) 
        filename = os.path.join(self.path, f'graph_')
        np.savez(filename, x=x, a=a)

    def read(self):
        # We must return a list of Graph objects
        output = []
        data = np.load(os.path.join(self.path, f'graph_.npz'))
        output.append(
                Graph(x=data['x'], a=data['a']))
        
        return output
def graph_adj(edges):
    a1 = get_edge_indices(edges,1)
    a2 = get_edge_indices(edges,2)
    a3 = get_edge_indices(edges,3)
    m1=Adjacency(a1,edges.shape[1])
    m2=Adjacency(a2,edges.shape[1])
    m3=Adjacency(a3,edges.shape[1])
    return [m1,m2,m3]

def Adjacency(graph,size):
    index = 0  #Index of the sublist
    matrix = [[0]*size for i in range(size)]
    for i in range(len(graph[0])):
        matrix[graph[0][i]][graph[1][i]]=1
    #print(matrix)
    matrix = np.array(matrix)
    return matrix
    
def get_edge_indices(edges,number):
    """Returns edge indices of the adjacency matrix"""
    return np.where(edges==number)
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

def load_data(data_path, prefix='train', size=None, padding=None, load_time=False):
    if not os.path.exists(data_path):
        raise ValueError(f"path '{data_path}' does not exist")

    # Load timeseries data.
    timeseries_file_pattern = os.path.join(data_path, f'{prefix}_timeseries*.npy')
    all_data = _load_files(timeseries_file_pattern, np.float32, padding=padding, pad_dims=(2,))

    # Load edge data.
    edge_file_pattern = os.path.join(data_path, f'{prefix}_edge*.npy')
    all_edges = _load_files(edge_file_pattern, np.int, padding, pad_dims=(1, 2))

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


def preprocess_data(data, seg_len=1, pred_steps=1, edge_type=1, ground_truth=True):
    time_series, edges = data[:2]
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
    # no of steps to predict aren't sent to the stack others are and the stack returns a stack of timeseries
    # [1:195,2:196,3:197,4:198,5:199]  
    time_segs_stack = stack_time_series(time_series[:, :stop, :, :],
                                        seg_len)#stop=-1
    
    time_segs = time_segs_stack.reshape([-1, seg_len, num_nodes, ndims])#195,5,5,4, changes the timesegs to
    # [0]= first 5 timesteps,[1]= 2:6 timesteps etc
    
    if ground_truth:
        # Stack shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_nodes, ndims]
        expected_time_segs_stack = stack_time_series(time_series[:, seg_len:, :, :],
                                                     pred_steps)#shape=(1,195,5,4),[0]=5th timestep
        # same shape of expected_time_segs_stack and time_segs_stack
        assert (time_segs_stack.shape[1] == expected_time_segs_stack.shape[1]
                == time_steps - seg_len - pred_steps + 1)
        expected_time_segs = expected_time_segs_stack.reshape([-1, pred_steps, num_nodes, ndims])#195,5,5,4, changes the timesegs to
    # [0]=  5:10 timesteps,[1]= 6:11 timesteps etc
    else:
        expected_time_segs = None

    edges_one_hot = one_hot(edges, edge_label, np.float32)# first in case of 0 1 in place of pos:0,in case of 1,1 in place of pos:1 etc 1 matrix of 1 row 
    edges_one_hot = np.repeat(edges_one_hot, time_segs_stack.shape[1], axis=0)#shape changed to (195,5,5,4)
    edges = np.repeat(edges, time_segs_stack.shape[1], axis=0)
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

    return [time_segs, edges], expected_time_segs

def off_diag_matrix(n):
    return np.ones((n, n)) - np.eye(n)


def one_hot(labels, num_classes, dtype=np.int):
    identity = np.eye(num_classes, dtype=dtype)
    one_hots = identity[labels.reshape(-1)]
    return one_hots.reshape(labels.shape + (num_classes,))


def load_model(model, log_dir):
    checkpoint = os.path.join(log_dir, 'weights_spektral.h5')
    if os.path.exists(checkpoint):
        model.load_weights(checkpoint)


def save_model(model, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, 'weights_spektral.h5')

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

def _load_files(file_pattern, dtype, padding=None, pad_dims=None):
    files = sorted(glob.glob(file_pattern))
    #print(files)
    if not files:
        raise FileNotFoundError(f"no files matching pattern {file_pattern} found")

    all_data = []
    for f in files:
        data = np.load(f).astype(dtype)

        if padding is not None and pad_dims is not None:
            pad_shape = [(0, padding - s if i in pad_dims else 0) for i, s in enumerate(data.shape)]
            data = np.pad(data, pad_shape, mode='constant', constant_values=0)

        all_data.append(data)

    return np.concatenate(all_data, axis=0)

def stack_time_series(time_series, seg_len, axis=2):
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    time_steps = time_series.shape[1]
    return np.stack([time_series[:, i:time_steps+1-seg_len+i, :, :] for i in range(seg_len)],
                    axis=axis)#0:195,1:196,2:197,3:198,4:199 stacking in axis=2 in case of seg_len=5
def load_edge_data(data_path, prefix='train', size=None, padding=None, load_time=False):
    if not os.path.exists(data_path):
        raise ValueError(f"path '{data_path}' does not exist")
    
    edge_file_pattern = os.path.join(data_path, f'{prefix}_edge*.npy')
    all_edges = _load_files(edge_file_pattern, np.int, padding, pad_dims=(1, 2))
    all_edges = all_edges[0]

    return all_edges

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
        
        # tf.print(node_msgs)

        return node_msgs

