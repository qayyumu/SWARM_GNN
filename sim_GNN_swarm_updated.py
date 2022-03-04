### Simulation of graph neural network for swarm simulation
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

from GNN_swarm import *
from GNN_sw_util import *
from data_to_gif_file import environmentsetup, environmentsetup_new


def eval_baseline(eval_data): #  evalute baseline
    time_segs = eval_data[0]
    return np.mean(np.square(time_segs[:, :-1, :, :] -
                             time_segs[:, 1:, :, :]))

def main():
    if ARGS.train:
        prefix = 'train'   # training
    elif ARGS.eval:
        prefix = 'valid'
    else:
        prefix = 'test'

    logdir  = "logs/{}".format(time.time())
    model_params = load_model_params(ARGS.config)
    if ARGS.learning_rate is not None:
        model_params['learning_rate'] = ARGS.learning_rate

    # data contains edge_types if `edge=True`.
    print(f"No of simulations to load : {ARGS.no_of_sim}\n")
    print(f"No of epochs per simulation : {ARGS.epochs}\n\n")
    data = load_data(ARGS.data_dir,ARGS.no_of_sim,prefix=prefix,suffix=ARGS.suffix,more_sim=ARGS.more_sim,
                                 size=ARGS.data_size, padding=ARGS.max_padding)
    # data = load_data(ARGS.data_dir,ARGS.no_of_sim,prefix="",suffix=ARGS.suffix,more_sim=ARGS.more_sim,
    #                             size=ARGS.data_size, padding=ARGS.max_padding)
    for j in range(ARGS.hepochs):
        
        for i in range(ARGS.no_of_sim):
            print(f"EPOCH NO: {j}\n\n\n")
            # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
            if ARGS.no_of_sim == 1:

                input_data, expected_time_segs = preprocess_data(
                    [data[i],data[1]], model_params['time_seg_len'], ARGS.pred_steps_rem, edge_type=model_params['edge_type'], ground_truth=not ARGS.test)
            else:
                input_data, expected_time_segs = preprocess_data(
                    [data[0][i],data[1]], model_params['time_seg_len'], ARGS.pred_steps_rem, edge_type=model_params['edge_type'], ground_truth=not ARGS.test)
            print(f"\n{prefix.capitalize()} data from {ARGS.data_dir}, simulation no {i} processed.\n")

            nagents, ndims = data[0][i].shape[-2:]
            
            model = SwarmNet.build_model(nagents, ndims, model_params, logdir,1)
            model.summary()

            load_model(model, ARGS.log_dir)

            if ARGS.train:
                checkpoint = save_model(model, ARGS.log_dir)

                # Freeze some of the layers according to train mode.
                if ARGS.train_mode == 1:
                    model.conv1d.trainable = True

                    model.graph_conv.edge_encoder.trainable = True
                    model.graph_conv.node_decoder.trainable = False

                elif ARGS.train_mode == 2:
                    model.conv1d.trainable = False

                    model.graph_conv.edge_encoder.trainable = False
                    model.graph_conv.node_decoder.trainable = True

               
                #tb_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq=5,)
                model.fit(input_data, expected_time_segs,
                        epochs=ARGS.epochs, batch_size=ARGS.batch_size,
                        callbacks=[checkpoint]) #,CustomCallback()]),tb_callback

            elif ARGS.eval:
                result = model.evaluate(
                    input_data, expected_time_segs, batch_size=ARGS.batch_size)
                # result = MSE
                baseline = eval_baseline(data)
                print('Baseline:', baseline, '\t| MSE / Baseline:', result / baseline)

            elif ARGS.test:
                # Predicting 1 step and passing it to model again to make multiple predictions
                print('Prediction Started')
                total_steps=input_data[0].shape[0]
                red_data=data[0][0][:total_steps]# taking only the reduced number of steps
                prediction = model.predict(input_data)
                red_data=np.append(red_data,prediction[-1],axis=0)# appending the made prediction to prev data
                print(red_data.shape)
                print(prediction[-1])
                #print(predicted.shape)
                for i in range(ARGS.pred_steps-1):
                    #environmentsetup((f'prediction_{ARGS.pred_steps}'),4,f'test_{ARGS.pred_steps}',ARGS.pred_steps)
                    input_data=for_more_predictions(red_data,input_data[1],prediction)
                    print(input_data[0].shape)
                    prediction = model.predict(input_data)
                    red_data=np.append(red_data,prediction[-1],axis=0)# appending the made prediction to prev data
                    print(f"Data Shape Changed to: {red_data.shape}")
                    print(prediction[-1])
                    print(f'prediction No:{i+2} completed')
                print('After Prediction data shape :')
                print(red_data.shape)
                red_data=np.expand_dims(red_data,axis=0)
                np.save(os.path.join(ARGS.log_dir,
                       f'prediction_{ARGS.pred_steps}.npy'), red_data)
                 #no of boids = 4 here
                environmentsetup((f'prediction_{ARGS.pred_steps}'),4,f'test_{ARGS.pred_steps}',ARGS.pred_steps)
                print('Predictions saved in gif file')

def for_more_predictions(data,edges,prediction):
    """
    changing data to the req input format
    """
    predicted = data
    #predicted=np.append(top,prediction[-1],axis=0)
    predicted = np.expand_dims(predicted,axis=0)
    time_segs_stack = stack_time_series(predicted[:, :, :, :],5)
    time_steps, num_nodes, ndims = predicted.shape[1:]
    time_segs = time_segs_stack.reshape([-1, 5, num_nodes, ndims])
    #edges=np.append(edges,np.expand_dims(edges[0],axis=0),axis=0)
    # making edge matrix size = time segs size
    if time_segs.shape[0]<=edges.shape[0]:
        edges=edges[:(time_segs.shape[0])]
    else:
        i=time_segs.shape[0]-edges.shape[0]
        for _ in range(i):
            edges=np.append(edges,np.expand_dims(edges[0],axis=0),axis=0)


    return [time_segs,edges]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## default arguments is for train, current_directories, pred_size = 200
    parser.add_argument('--data-dir', type=str,default="data/",
                        help='data directory')
    parser.add_argument('--data-size', type=int, default=None,
                        help='optional data size cap to use for training')
    parser.add_argument('--config', type=str,default='example.json',
                        help='model config file')
    parser.add_argument('--log-dir', type=str,default='.',
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training steps')
    parser.add_argument('--hepochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--pred-steps_rem', type=int, default=990,
                        help='number of steps to remove from time series')
    parser.add_argument('--pred-steps', type=int, default=100,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--learning-rate', '--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--train-mode', type=int, default=0,
                        help='train mode determines which layers are frozen: '
                             '0 - all layers are trainable; '
                             '1 - conv1d layers and edge encoders are trainable; '
                             '2 - edge encoders and node encoder are trainable.')
    parser.add_argument('--max-padding', type=int, default=None,
                        help='max pad length to the number of agents dimension')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='turn on evaluation')
    parser.add_argument('--test', action='store_true', default=True,
                        help='turn on test')
    parser.add_argument('--more-sim', action='store_true', default=False,
                        help='if more than 1 simulation data to load ')
    parser.add_argument('--no-of-sim', type=int, default=1,
                        help='no of simulations to load')
    parser.add_argument('--suffix', type=str, default='savedata_',
                        help='suffix for save files')
    
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()

#python run_swarwmnet.py --data-dir path/to/training/data --log-dir path/to/log/dir --config path/to/config/file --pred-steps <prediction_horizon> --train --epochs <num_epochs>
#tensorboard --logdir=logs