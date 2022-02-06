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
        self.time_seg_len= 5
        self.edge_type = model_params['edge_type'] 
        # Layers
        self.conv1d = utils.Conv1D(model_params['cnn']['filters'], name='Conv1D')
        self.encoder1 = GCSConv(64,activation="tanh")#
        self.encoder2 = GCSConv(64,activation="tanh")#,activation="tanh"
        self.encoder3 = GCSConv(64,activation="tanh")#,activation="tanh"
        self.aggregator = keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=[2]))
        self.aggregator = keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=[2])) # aggregator for messages from different edge types
 
        self.decoder= Dense(64,activation="relu") # decoder
        self.outlayer = Dense(output_dim,activation="tanh") # get in the form of (posx,posy,velx,vely)
        self.activation = keras.layers.Activation('tanh')
        


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
        condensed_state = self.conv1d(time_segs)
        #condensed_state shape [batch, num_nodes, 1, filters]

        condensed_state = tf.squeeze(condensed_state, axis=2)
        # condensed_state shape [batch, num_nodes, filters]
        
        # load edges 
        edges=utils.load_edge_data("Data",
                                   prefix='train', size=None, padding=None)
        # turn the hetrogeneous graph into  edge types graphs
        all_edges_types=utils.graph_adj(edges)
        
        # turn edge matrix into tensor
        for i in range(self.edge_type):
            all_edges_types[i]==tf.convert_to_tensor(all_edges_types[i],dtype=float)
        
        encoded_msgs_by_type = []
        # find encoding messages
        #for i in range(self.edge_type):
        #     encoded_msgs = self.encoders[i](condensed_state,all_edges_types[i])
        encoded_msgs = self.encoder1([condensed_state,all_edges_types[0]])    
        encoded_msgs_by_type.append(encoded_msgs)
        encoded_msgs = self.encoder2([condensed_state,all_edges_types[1]])    
        encoded_msgs_by_type.append(encoded_msgs)
        encoded_msgs = self.encoder3([condensed_state,all_edges_types[2]])    
        encoded_msgs_by_type.append(encoded_msgs)
        
        encoded_msgs_by_type = tf.stack(encoded_msgs_by_type, axis=2)
        encoded  = self.activation(self.aggregator(encoded_msgs_by_type)) # aggregating messages from all the graph types

        # finding nodes with no incoming edge
        #e= tf.reduce_sum(edges,axis=0)
        e= np.sum(edges,axis=0)
        b= np.where(e>0)
        # Convert array to give weight=1 to nodes having edges
        for i in b:
            e[i]=1
        e=tf.expand_dims(e, -1)
        e=tf.cast(e, tf.float32)
        # Multiplying embeddings with e to get no edge msg nodes to 0
        encoded_states= tf.multiply(encoded,
                                e)

        node_states= self.decoder(tf.concat([condensed_state,encoded_states],axis=-1))

        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1, :] 

        next_state = prev_state + self.outlayer(node_states)

        return next_state
      

    @classmethod
    def build_model(cls, num_nodes, output_dim, model_params, pred_steps=1, return_inputs=False):
        model = cls(num_nodes, output_dim, model_params, pred_steps)

        optimizer = keras.optimizers.Adam(learning_rate=model_params['learning_rate'])

        model.compile(optimizer, loss='mse')

        input_shape = [(None, 5, num_nodes, output_dim),
                       (num_nodes, num_nodes)]

        inputs = model.build(input_shape)

        if return_inputs:
            return model, inputs

        return model