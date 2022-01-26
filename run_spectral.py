import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils_spektral as utils
import swarmnet_spectral as swarmnet

def main():
    
    if ARGS.train:
        prefix = 'train'
    else:
        prefix = 'test'

    model_params = utils.load_model_params(ARGS.config)
    if ARGS.learning_rate is not None:
        model_params['learning_rate'] = ARGS.learning_rate

    #data[0]=timeseries,data[1]=edges
    data = utils.load_data(ARGS.data_dir,
                                   prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding)

    # input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
    print('After load data')
    input_data, expected_time_segs = utils.preprocess_data(
        data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'], ground_truth=not ARGS.test)
    print(f"\n{prefix.capitalize()} data from {ARGS.data_dir} processed.\n")
    

    nagents, ndims = data[0].shape[-2:]#nagents= biods+viscos+goal+obstacle,ndims=4(position[2],velocity[2])
    #model = swarmnet.SwarmNet(nagents,ndims,model_params,ARGS.pred_steps)
    model = swarmnet.SwarmNet.build_model(nagents, ndims,model_params, ARGS.pred_steps)
    model.summary()

    utils.load_model(model, ARGS.log_dir)
    
    if ARGS.train:
        checkpoint = utils.save_model(model, ARGS.log_dir)
        model.fit(input_data, expected_time_segs,
                  epochs=ARGS.epochs, batch_size=ARGS.batch_size,
                  callbacks=[checkpoint])

    elif ARGS.test:
        prediction = model.predict(input_data)
        np.save(os.path.join(ARGS.log_dir,
                f'prediction_{ARGS.pred_steps}.npy'), prediction)


if __name__ == '__main__':
    class EmptyClass():
        pass
    ARGS = EmptyClass()
    ARGS.data_dir = "data"
    ARGS.data_size=None
    ARGS.config='example.json'
    ARGS.log_dir='.'
    ARGS.epochs=100
    ARGS.pred_steps=100
    ARGS.batch_size=64
    ARGS.learning_rate=0.0001
    ARGS.train=True
    ARGS.train_mode=0
    ARGS.max_padding=None
    ARGS.eval=False
    ARGS.test=False

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()
