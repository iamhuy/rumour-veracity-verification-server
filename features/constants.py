brown_cluster_dict_filename = "brown-cluster.model"

DATASET_NAME = 'semeval2017-task8-dataset'
TESTSET_NAME = 'semeval2017-task8-test-data'

RAW_OUTPUT_FOLDER = 'traindev/'
RAW_INPUT_FOLDER = 'rumoureval-data/'
DATASET_FULL_EVENTS = ['charliehebdo/', 'ebola-essien/', 'ferguson/', 'germanwings-crash/', 'ottawashooting/', 'prince-toronto/', 'putinmissing/', 'sydneysiege/']
DATASET_EVENTS = ['charliehebdo/', 'ferguson/', 'germanwings-crash/', 'ottawashooting/', 'sydneysiege/']
VERACITY_LABEL_FILE = ['rumoureval-subtaskB-dev.json', 'rumoureval-subtaskB-train.json']
STANCE_LABEL_FILE = ['rumoureval-subtaskA-dev.json', 'rumoureval-subtaskA-train.json']
VERACITY_LABEL_TEST_FILE = ['subtaskB.json']
STANCE_LABEL_TEST_FILE = ['subtaskA.json']

DATASET_OUT = 'make_dataset'
VERACITY_LABELS = ["true", "false", "unverified"]
VERACITY_LABELS_MAPPING = {
    "true": 0,
    "false": 1,
    "unverified":2
}

STANCE_LABELS = ["comment", "deny", "support", "question"]
STANCE_LABELS_MAPPING = {
    "comment":0,
    "deny": 1,
    "support": 2,
    "question": 3
}