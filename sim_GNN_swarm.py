### Simulation of graph neural network for swarm simulation
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

from GNN_swarm import *
from GNN_sw_util import *


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

    model_params = load_model_params(ARGS.config)
    if ARGS.learning_rate is not None:
        model_params['learning_rate'] = ARGS.learning_rate

    # data contains edge_types if `edge=True`.
    data = load_data(ARGS.data_dir,
                                   prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding)

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    input_data, expected_time_segs = preprocess_data(
        data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'], ground_truth=not ARGS.test)
    print(f"\n{prefix.capitalize()} data from {ARGS.data_dir} processed.\n")

    nagents, ndims = data[0].shape[-2:]

    model = SwarmNet.build_model(nagents, ndims, model_params, ARGS.pred_steps)
    # model.summary()

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

        model.fit(input_data, expected_time_segs,
                  epochs=ARGS.epochs, batch_size=ARGS.batch_size,
                  callbacks=[checkpoint])

    elif ARGS.eval:
        result = model.evaluate(
            input_data, expected_time_segs, batch_size=ARGS.batch_size)
        # result = MSE
        baseline = eval_baseline(data)
        print('Baseline:', baseline, '\t| MSE / Baseline:', result / baseline)

    elif ARGS.test:
        prediction = model.predict(input_data)
        np.save(os.path.join(ARGS.log_dir,
                f'prediction_{ARGS.pred_steps}.npy'), prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-size', type=int, default=None,
                        help='optional data size cap to use for training')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
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
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()
