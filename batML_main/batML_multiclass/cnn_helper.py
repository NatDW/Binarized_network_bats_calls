import numpy as np
import tensorflow as tf
from larq_zoo.literature import BinaryResNetE18
from larq.layers import QuantConv2D, QuantDense
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from sklearn.utils import class_weight
from hyperopt import hp, tpe, fmin, space_eval, Trials
import pickle
import gc

from batML_main.batML_multiclass.data_set_params import DataSetParams
from batML_main.batML_multiclass.models_params_helper import params_to_dict

def build_cnn(params, ip_size, nb_output, nb_cnn):
    """
    Build a Convolutional Neural Network with the specified parameters.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    ip_size : numpy array
        Dimention of the input to the CNN.
    nb_output : int
        Number of output nodes.
    nb_cnn : String
        "_1" or "_2" if the double cnn architecture and "" otherwise.

    Returns
    --------
    net : tensorflow.keras.models.Sequential
        Sequential CNN.
    """

    nb_conv_layers = getattr(params, "nb_conv_layers"+nb_cnn)
    nb_dense_layers = getattr(params, "nb_dense_layers"+nb_cnn)
    nb_filters = getattr(params, "nb_filters"+nb_cnn)
    filter_size = getattr(params, "filter_size"+nb_cnn)
    pool_size = getattr(params, "pool_size"+nb_cnn)
    nb_dense_nodes = getattr(params, "nb_dense_nodes"+nb_cnn)
    dropout_proba = getattr(params, "dropout_proba"+nb_cnn)
    kwargs = dict(use_bias=False, input_quantizer="ste_sign", kernel_quantizer="ste_sign",
                                    kernel_constraint="weight_clip")
    #kwargs = dict(use_bias=False, input_quantizer=None, kernel_quantizer=None, kernel_constraint=None)

    net = tf.keras.models.Sequential()
    net.add(QuantConv2D(nb_filters, (filter_size,filter_size), padding="same",
                                        activation='linear', input_shape=(ip_size[0], ip_size[1], 1),
                                        kernel_quantizer="ste_sign", kernel_constraint="weight_clip", use_bias=False))
    net.add(BatchNormalization(momentum=0.9))
    net.add(MaxPool2D(pool_size=(pool_size, pool_size)))
    #net.add(BatchNormalization(momentum=0.9))
    for i in range(nb_conv_layers):
        net.add(QuantConv2D(nb_filters, (filter_size,filter_size), padding="same", activation='linear', **kwargs))
        net.add(BatchNormalization(momentum=0.9))
        net.add(MaxPool2D(pool_size=(pool_size, pool_size)))
        #net.add(BatchNormalization(momentum=0.9))
    #net.add(Dropout(0.3))
    net.add(Flatten())
    for i in range(nb_dense_layers):
        net.add(QuantDense(nb_dense_nodes, activation='linear', **kwargs))
        net.add(BatchNormalization(momentum=0.9))
        #net.add(Dropout(0.3))
    net.add(QuantDense(nb_output, activation='linear', **kwargs))
    net.add(BatchNormalization(momentum=0.9))
    net.add(Activation('softmax'))
    #net = BinaryResNetE18(input_shape=(ip_size[0], ip_size[1], 1), weights=None, num_classes=nb_output)
    net.summary()
    return net

def network_fit(params, features, labels, nb_output, nb_cnn=''):
    """
    Build and fit the Convolutional Neural Network.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    features : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : numpy array
        Class label (0-7) for each training position.
    nb_output : int
        Number of output nodes.
    nb_cnn : String
        "_1" or "_2" if the double cnn architecture and "" otherwise.
    
    Returns
    --------
    network : tensorflow.keras.models.Sequential
        Fit CNN.
    history : list
        History of the monitored metrics for each epoch.
    """
    
    tf.keras.backend.clear_session()
    gc.collect()

    print("CNN params= ", params_to_dict(params))
    learn_rate_adam = getattr(params, "learn_rate_adam"+nb_cnn)
    beta_1 = getattr(params, "beta_1"+nb_cnn)
    beta_2 = getattr(params, "beta_2"+nb_cnn)
    epsilon = getattr(params, "epsilon"+nb_cnn)
    min_delta = getattr(params, "min_delta"+nb_cnn)
    patience = getattr(params, "patience"+nb_cnn)
    batchsize = getattr(params, "batchsize"+nb_cnn)

    # Build the CNN
    network = build_cnn(params, features.shape[2:], nb_output, nb_cnn)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    opti = tf.keras.optimizers.Adam( learning_rate=learn_rate_adam, beta_1=beta_1, beta_2=beta_2,
                                    epsilon=epsilon, name="Adam")
    network.compile(optimizer=opti, loss=loss_fn, metrics=['accuracy', 'sparse_categorical_accuracy'])
    class_w = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience,
                                                verbose=1, restore_best_weights=params.restore_best_weights)
    features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
    
    # Fit the CNN
    print("Fit the CNN")
    history = network.fit( features, labels, epochs=params.num_epochs, batch_size=batchsize,
                                shuffle=True, verbose=2, class_weight=class_w,
                                validation_split=params.validation_split, callbacks=[callback])
    return network, history

def obj_func_cnn(args):
    """
    Fits and returns the best loss of a CNN with given parameters.

    Parameters
    -----------
    args : dict
        Dictionnary of all the parameters needed to fit a CNN.

    Returns
    --------
    min_loss : float
        minimum value of the loss during training of the CNN.
    """
    
    params_cnn = DataSetParams()
    # CNN
    params_cnn.nb_conv_layers = args['nb_conv_layers']
    params_cnn.nb_dense_layers = args['nb_dense_layers']
    params_cnn.nb_filters = args['nb_filters']
    params_cnn.filter_size = args['filter_size']
    params_cnn.pool_size = args['pool_size']
    params_cnn.nb_dense_nodes = args['nb_dense_nodes']
    params_cnn.dropout_proba = args['dropout_proba']
    #Adam
    params_cnn.learn_rate_adam = args['learn_rate_adam']
    params_cnn.beta_1 = args['beta_1']
    params_cnn.beta_2 = args['beta_2']
    params_cnn.epsilon = args['epsilon']
    # early stopping
    params_cnn.min_delta = args['min_delta']
    params_cnn.patience = args['patience']
    # fit
    params_cnn.batchsize = args['batchsize']

    _, history = network_fit(params_cnn, args['features'], args['labels'], args['nb_output'])
    min_loss = np.min(history.history['val_loss'])
    return min_loss

def tune_network(params, features, labels, trials_filename, goal=None):
    """
    Tunes the network with hyperopt.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    features : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : numpy array
        Class label (0-7) for each training position.
    trials_filename : String
        Name of the file where the previous iterations of hyperopt are saved.
    goal : String
        Indicates whether the network needs to be tuned for detection or classification.
        Can be either "detection" or "classification".
    """

    print("\n tune cnn")
    nb_output = 8
    if goal == "detection":
        nb_output = 2
    elif goal == "classification": 
        nb_output = 7
    space_cnn = {  'nb_conv_layers': hp.choice('nb_conv_layers', range(1,4)),
                    'nb_dense_layers': hp.choice('nb_dense_layers', range(1,5)),
                    'nb_filters': hp.choice('nb_filters', range(16, 65, 8)),
                    'filter_size': hp.choice('filter_size', range(2,6)),
                    'pool_size': 2,
                    'nb_dense_nodes': hp.choice('nb_dense_nodes', range(64, 513, 64)),
                    'dropout_proba': hp.choice('dropout_proba', np.arange(0.3, 0.8, 0.1)),
                    'learn_rate_adam': hp.choice('learn_rate_adam', np.logspace(-5, -2, num=15)),
                    'beta_1': hp.choice('beta_1', [0.8, 0.9, 0.95]),
                    'beta_2': hp.choice('beta_2', [0.95, 0.999]),
                    'epsilon': hp.choice('epsilon', [1e-8]),
                    'min_delta': hp.choice('min_delta', [0.00005, 0.0005, 0.005]),
                    'patience': hp.choice('patience', [5, 10, 15, 20]),
                    'batchsize': hp.choice('batchsize', range(32, 128, 8)),
                    'features': features,
                    'labels': labels,
                    'nb_output': nb_output
                }
    
    # load the saved trials
    try:
        trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
        max_trials = len(trials.trials) + 1
    # create a new trials
    except:
        max_trials = 1
        trials = Trials()

    # optimise the objective function with the defined set of CNN parameters
    best_space_indices = fmin(obj_func_cnn, space_cnn, trials=trials, algo=tpe.suggest, max_evals=max_trials)
    best_space = space_eval(space_cnn, best_space_indices)
    best_space = {k: best_space[k] for k in best_space.keys() - {'features', 'labels'}}
    print("best_space=",best_space)
    with open(trials_filename + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    # CNN
    params.nb_conv_layers = best_space['nb_conv_layers']
    params.nb_dense_layers = best_space['nb_dense_layers']
    params.nb_filters = best_space['nb_filters']
    params.filter_size = best_space['filter_size']
    params.pool_size = 2
    params.nb_dense_nodes = best_space['nb_dense_nodes']
    params.dropout_proba = best_space['dropout_proba']
    # Adam
    params.learn_rate_adam = best_space['learn_rate_adam']
    params.beta_1 = best_space['beta_1']
    params.beta_2 = best_space['beta_2']
    params.epsilon = best_space['epsilon']
    # early stopping
    params.min_delta = best_space['min_delta']
    params.patience = best_space['patience']
    # fit
    params.batchsize = best_space['batchsize']
