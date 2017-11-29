import json
import pickle
import os
from settings import MODELS_ROOT
from utils import get_subset_features

feature_bitmask_file = open(os.path.join(MODELS_ROOT, 'feature_bitmask'), "r")
feature_bitmask = json.loads(feature_bitmask_file.read())

reducer = None
scaler = None
balancer = None

reducer = None
scaler = None
balancer = None
feature_selector = None

settings = pickle.load(open(os.path.join(MODELS_ROOT, 'settings.model'), "rb"))
# X_test = get_subset_features(X_test, feature_option=settings['features_subset'])

if settings['feature_selection_algorithm'] != None:
    feature_selector = pickle.load(open(os.path.join(MODELS_ROOT, 'feature_selector.model'), "rb"))

classifier = pickle.load(open(os.path.join(MODELS_ROOT, 'classifier.model'), "rb"))


if settings['scale_option'] != None:
    scaler = pickle.load(open(os.path.join(MODELS_ROOT, 'scaler.model'), "rb"))

if settings['reduce_dimension_algorithm'] != None:
    reducer = pickle.load(open(os.path.join(MODELS_ROOT, 'reducer.model'), "rb"))


