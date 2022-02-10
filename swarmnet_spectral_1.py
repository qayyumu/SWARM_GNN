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
        self.nodepropogator = utils.NodePropagator() 
        # Layers
        self.conv1d = utils.Conv1D(model_params['cnn']['filters'], name='Conv1D')
        self.encoder = GCSConv(64,activation="tanh")#
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
        
        nodes=self.nodepropogator(condensed_state)
        # load edges 
        edges=utils.load_edge_data("Data",
                                   prefix='train', size=None, padding=None)
           
        edge=normalized_adjacency(edges,symmetric=False)
        edge=tf.convert_to_tensor(edge,dtype=float)

        e= np.sum(edges,axis=0)
        #nodes=tf.transpose(nodes,[1,0,2,3])
        gcnresult=[]
        j=0
        for i in e:
            if i != 0:
                nodefeature=nodes[:,j,:,:]
                # to get the self loop msg to 0
                ones=np.ones(len(e))
                ones[j]=0
                noselfedge=np.expand_dims(ones,-1)
                nodefeature=tf.multiply(nodefeature,noselfedge)
                ### Apply GCN Layer
                nodefeature=self.encoder(nodefeature,edge)# <tf.Tensor 'Squeeze:0' shape=(None, 7, 64) dtype=float32>
                #<tf.Tensor 'Mul_3:0' shape=(None, 7, 128) dtype=float32>
                # Batch,No of Nodes,64
                gcnresult.append(nodefeature[:,j,:])

            else:
                # Nodes with 0 inedge make it zero
                nodefeature=nodes[:,j,:,:]
                zeros=np.zeros(len(e))
                e=np.expand_dims(zeros,-1)
                nodefeature=tf.multiply(nodefeature,e)
                gcnresult.append(nodefeature[:,j,:])
            j=j+1


        node_states= self.decoder(tf.concat([condensed_state,gcnresult],axis=-1))

        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1, :] 

        next_state = prev_state + self.outlayer(node_states)

        return next_state
      

    @classmethod
    def build_model(cls, num_nodes, output_dim, model_params, pred_steps=1, return_inputs=False):
        model = cls(num_nodes, output_dim, model_params, pred_steps)

        optimizer = keras.optimizers.Adam(learning_rate=model_params['learning_rate'])

        model.compile(optimizer,run_eagerly=True, loss='mse')

        input_shape = [(None, 5, num_nodes, output_dim),
                       (num_nodes, num_nodes)]

        inputs = model.build(input_shape)

        if return_inputs:
            return model, inputs

        return model