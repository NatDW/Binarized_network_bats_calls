import numpy as np
import pickle
import os
import time
import tensorflow as tf
import larq
import json
import batML_main.batML_multiclass.evaluate as evl
import joblib
from network_builder import quantize_network as quant
from batML_main.batML_multiclass.data_set_params import DataSetParams
from tensorflow.keras.models import load_model, Model
import batML_main.batML_multiclass.classifier as clss
import xgboost as xgb


def load_data(data_set, goal):
    loaded_data_tr = np.load(data_set, fix_imports=True, allow_pickle=True, encoding='latin1')
    train_pos = loaded_data_tr['train_pos']
    train_files = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    train_classes = []
    test_pos = loaded_data_tr['test_pos']
    test_files = loaded_data_tr['test_files']
    test_durations = loaded_data_tr['test_durations']
    test_classes = []

    # Some datasets like Batdetective's put an additional axis around each position while others do not.
    # We uniformise by adding the axis when it is not present.
    type_train = None
    type_test = None
    i = 0
    j = 0
    while type_train is None:
        type_train = None if len(train_pos[i]) == 0 else type(train_pos[i][0])
        i += 1
    while type_test is None:
        type_test = None if len(test_pos[j]) == 0 else type(test_pos[j][0])
        j += 1
    if type_train != np.ndarray:
        for ii in range(len(train_pos)):
            train_pos[ii] = train_pos[ii][..., np.newaxis]
    if type_test != np.ndarray:
        for ii in range(len(test_pos)):
            test_pos[ii] = test_pos[ii][..., np.newaxis]

    if goal == "detection":
        train_files = np.array(list(map(lambda x: x.decode('ascii'), train_files)))
        test_files = np.array(list(map(lambda x: x.decode('ascii'), test_files)))
    elif goal == "classification":
        test_classes = loaded_data_tr['test_class']
        train_classes = loaded_data_tr['train_class']
        # Some datasets like Batdetective's put an additional axis around each class while others do not.
        # We uniformise by adding the axis when it is not present.
        type_train = None
        type_test = None
        i = 0
        j = 0
        while type_train is None:
            type_train = None if len(train_classes[i]) == 0 else type(train_classes[i][0])
            i += 1
        while type_test is None:
            type_test = None if len(test_classes[j]) == 0 else type(test_classes[j][0])
            j += 1
        if type_train != np.ndarray:
            for ii in range(len(train_classes)):
                train_classes[ii] = train_classes[ii][..., np.newaxis]
        if type_test != np.ndarray:
            for ii in range(len(test_classes)):
                test_classes[ii] = test_classes[ii][..., np.newaxis]
    print("train size", goal, " = ", train_files.shape)
    print("test size", goal, " = ", test_files.shape)
    return train_pos, train_files, train_durations, train_classes, test_pos, test_files, test_durations, test_classes


if __name__ == "__main__":
    on_GPU = True
    load_features_from_file = False
    model_name = "hybrid_cnn_xgboost"  # can be one of: 'batmen', 'cnn2',  'hybrid_cnn_svm', 'hybrid_cnn_xgboost', 'hybrid_call_svm', 'hybrid_call_xgboost'
    result_dir = 'results/'  # where we will store the outputs
    model_dir = '/home/ndewinter/code/batML_main/batML_multiclass/data/models/'

    test_set = 'Natagora'
    data_set_test = '/home/ndewinter/data/train_test_split/test_set_' + test_set + '.npz'
    raw_audio_dir_detect = '/home/ndewinter/data/wav/'
    raw_audio_dir_classif = '/storage/wav/'

    if on_GPU:
        # needed to run tensorflow on GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        # needed to run tensorflow on CPU
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        tf.config.set_visible_devices([], 'GPU')
        session = tf.compat.v1.InteractiveSession(config=config)

    _, _, _, _, pos_test, files_test, durations_test, classes_test = load_data(data_set_test, "classification")

    # model name and load models
    """if model_name == "batmen":
        date = "23_04_21_09_37_13_"
        hnm = "_hnm1"
        model_file_classif = model_dir + date + "classif_" + model_name + hnm
        network_classif = load_model(model_file_classif + '_model')"""
    if model_name == "batmen":
        date = "08_12_21_20_30_11_"
        hnm = "_hnm0"
        model_file_classif = model_dir + date + "classif_" + "batmen" + hnm
        network_classif = load_model(model_file_classif + "_model")
    elif model_name == "cnn2":
        date = "23_04_21_14_51_55_"
        hnm = "_hnm0"
        model_file_detect = model_dir + date + "detect_" + model_name + hnm
        network_detect = load_model(model_file_detect + '_model')
        model_file_classif = model_dir + date + "classif_" + model_name + hnm
        network_classif = load_model(model_file_classif + '_model')
    elif model_name == "hybrid_cnn_svm":
        date = "10_05_21_08_40_17_"
        hnm = "_hnm0"
        model_file_features = model_dir + date + "features_" + model_name + hnm
        network_features = load_model(model_file_features + '_model')
        network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
        model_file_classif = model_dir + date + "classif_" + model_name + hnm
        network_classif = joblib.load(model_file_classif + '_model.pkl')
        scaler = joblib.load(model_file_classif + '_scaler.pkl')
    elif model_name == "hybrid_cnn_xgboost":
        date = "03_05_21_16_11_14_"
        hnm = "_hnm0"
        model_file_features = model_dir + date + "features_" + model_name + hnm
        network_features = load_model(model_file_features + '_model')
        network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
        model_file_classif = model_dir + date + "classif_" + model_name + hnm
        network_classif = xgb.XGBClassifier()
        network_classif.load_model(model_file_classif + '_model.json')
    elif model_name == "hybrid_call_svm":
        date = "23_04_21_14_24_46_"
        hnm = "_hnm0"
        model_file_detect = model_dir + date + "detect_" + model_name
        network_detect = load_model(model_file_detect + '_model')
        model_file_classif = model_dir + date + "classif_" + model_name
        network_classif = joblib.load(model_file_classif + '_model.pkl')
        scaler = joblib.load(model_file_classif + '_scaler.pkl')
    elif model_name == "hybrid_call_xgboost":
        date = "23_04_21_13_51_38_"
        hnm = "_hnm0"
        model_file_detect = model_dir + date + "detect_" + model_name + hnm
        network_detect = load_model(model_file_detect + '_model')
        model_file_classif = model_dir + date + "classif_" + model_name + hnm
        network_classif = xgb.XGBClassifier()
        network_classif.load_model(model_file_classif + '_model.json')

    # load params
    with open(model_file_classif + '_params.p') as f:
        parameters = json.load(f)
    print("params=", parameters)

    # array with group name according to class number
    group_names = ['not call', 'Barbarg', 'Envsp', 'Myosp', 'Pip35', ' Pip50', 'Plesp', 'Rhisp']

    # model classifier
    params = DataSetParams(model_name)
    params.window_size = parameters['win_size']
    params.max_freq = parameters['max_freq']
    params.min_freq = parameters['min_freq']
    params.mean_log_mag = parameters['mean_log_mag']
    params.fft_win_length = parameters['slice_scale']
    params.fft_overlap = parameters['overlap']
    params.crop_spec = parameters['crop_spec']
    params.denoise = parameters['denoise']
    params.smooth_spec = parameters['smooth_spec']
    params.nms_win_size = parameters['nms_win_size']
    params.smooth_op_prediction_sigma = parameters['smooth_op_prediction_sigma']
    if model_name in ["hybrid_cnn_xgboost", "hybrid_call_xgboost"]: params.n_estimators = parameters["n_estimators"]
    params.load_features_from_file = load_features_from_file
    params.detect_time = 0
    params.classif_time = 0
    params.audio_dir_detect = raw_audio_dir_detect
    params.audio_dir_classif = raw_audio_dir_classif
    model_cls = clss.Classifier(params)

    if model_name in ["batmen", "cnn2", "hybrid_cnn_svm", "hybrid_cnn_xgboost", "hybrid_call_svm",
                      "hybrid_call_xgboost"]:
        model_cls.model.network_classif = network_classif
    if model_name in ["cnn2", "hybrid_call_svm", "hybrid_call_xgboost"]:
        model_cls.model.network_detect = network_detect
    if model_name in ["hybrid_cnn_svm", "hybrid_cnn_xgboost"]:
        model_cls.model.network_features = network_features
        model_cls.model.model_feat = network_feat
    if model_name in ["hybrid_cnn_svm", "hybrid_call_svm"]:
        model_cls.model.scaler = scaler

    nms_pos_test, nms_prob_test, pred_classes_test, nb_windows_test = model_cls.test_batch("classification", files_test,
                                                                                           durations_test)
    evl.prec_recall_1d(nms_pos_test, nms_prob_test, pos_test, pred_classes_test, classes_test,
                       durations_test, model_cls.params.detection_overlap, model_cls.params.window_size,
                       nb_windows_test,
                       model_dir + model_name + ".txt")
