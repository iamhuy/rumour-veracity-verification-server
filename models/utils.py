from settings import *
import json

feature_bitmask_file = open(os.path.join(MODELS_ROOT, 'feature_bitmask'), "r")
feature_bitmask = json.loads(feature_bitmask_file.read())


def get_subset_features(X, feature_option):
    """
        Get a subset of features from predefined settings
    :param feature_option: a string as a name of settings
            X: original features
    :return: subset of result features
    """
    X_new = []
    for instance in X:
        instance_new = []
        for feature_index in FEATURE_OPTIONS[feature_option]:
            start, end = feature_bitmask[FEATURE_LIST[feature_index]]
            instance_new += instance[start : end]
        X_new.append(instance_new)

    return X_new


def get_array(idx_array, x):
    return [x[i] for i in idx_array]
