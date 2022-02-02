from turtle import right
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GATConv,GCSConv
from spektral.utils import normalized_adjacency
import utils_spektral as utils


class SwarmNet(Model):
    def __init__(self, num_nodes, output_dim, model_params, pred_steps=1, name='SwarmNet'):
        super().__init__()
        #input(195,5,7,4),output(195,1,7,4)
        self.pred_steps=pred_steps
        self.time_seg_len= 1
        # Layers
        #self.conv1d = utils.Conv1D(model_params['cnn']['filters'], name='Conv1D')
        self.conv1 = GCSConv(4,activation="tanh")#
        self.conv2 = GCSConv(4,activation="tanh")#,activation="tanh"
        self.conv3 = GCSConv(4,activation="tanh")#,activation="tanh"
        self.aggregator = keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=[2]))
        self.activation = keras.layers.Activation('tanh')
        self.out= Dense(4,activation="tanh")

        #self.dense1= Dense(32,activation="relu")
        #self.dense = Dense(output_dim,activation="tanh")
        


    def call(self, inputs,training=False):
        time_segs,edges = inputs
        extended_time_segs = tf.transpose(time_segs, [0, 2, 1, 3])
        for i in range(self.pred_steps):
            next_state = self._pred_next(extended_time_segs[:, :, i:, :], edges,
                                         training=training)
            next_state = tf.expand_dims(next_state, axis=2)
            extended_time_segs = tf.concat([extended_time_segs, next_state], axis=2)

        # Transpose back to [batch, time_seg_len+pred_steps, num_agetns, ndims]
        extended_time_segs = tf.transpose(extended_time_segs, [0, 2, 1, 3])

        # Return only the predicted part of extended_time_segs
        return extended_time_segs[:, self.time_seg_len:, :, :]

    def build(self, input_shape):
        t = keras.layers.Input(input_shape[0][1:])
        e = keras.layers.Input(input_shape[1][1:])
        inputs = [t, e]

        self.call(inputs)
        self.built = True
        return inputs

    def _pred_next(self, time_segs, edges, training=False):
        X=time_segs
        X=tf.squeeze(X,axis=2)
        #print(X.shape)
        
        edges=utils.load_edge_data("Data",
                                   prefix='train', size=None, padding=None)
         
        edges=edges[0]
        edges1,edges2,edges3=utils.graph_adj(edges)
        #print(edges1)
        a1=tf.convert_to_tensor(edges1,dtype=float)
        a2=tf.convert_to_tensor(edges2,dtype=float)
        a3=tf.convert_to_tensor(edges3,dtype=float)

        #a1 = normalized_adjacency(edges1)
        #a2 = normalized_adjacency(edges2)
        #a3 = normalized_adjacency(edges3)
        x1 = tf.expand_dims(self.conv1([X, a1]), 2)
        x2 = tf.expand_dims(self.conv2([X, a2]), 2)
        x3 = tf.expand_dims(self.conv3([X, a3]), 2)
        
        x  = tf.concat((x1, x2, x3), axis=-2)# [batch,5, 3, 4]
        X  = self.activation(self.aggregator(x)) # aggregating messages from all the graph types
        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1, :]
        a=self.out(X)
        
        next_state = prev_state + a
        return next_state
      

    @classmethod
    def build_model(cls, num_nodes, output_dim, model_params, pred_steps=1, return_inputs=False):
        model = cls(num_nodes, output_dim, model_params, pred_steps)

        optimizer = keras.optimizers.Adam(learning_rate=model_params['learning_rate'])

        model.compile(optimizer, loss='mse')

        input_shape = [(None, 1, num_nodes, output_dim),
                       (num_nodes, num_nodes)]

        inputs = model.build(input_shape)

        if return_inputs:
            return model, inputs

        return model
