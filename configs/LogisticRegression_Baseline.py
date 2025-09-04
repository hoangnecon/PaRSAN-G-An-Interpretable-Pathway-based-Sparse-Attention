from configs import base_config as base

EXPERIMENT_NAME = "LogisticRegression_Baseline"
MODEL_NAME = 'LogisticRegression'
USE_SAMPLING = True
SAMPLING_METHOD = "BorderlineSMOTE"
MAX_LASSO_FEATURES = 2000
NUM_CLASSES = 4
N_SPLITS=10


X_TRAIN_FILENAME = 'X_train_pam50.pkl'
Y_TRAIN_FILENAME = 'y_train_pam50.pkl'
X_TEST_FILENAME = 'X_test_pam50.pkl'
Y_TEST_FILENAME = 'y_test_pam50.pkl'
PATHWAY_MAP_FILENAME = 'pathway_gene_mapping_pam50_integrated.json'
FEATURE_MAP_FILENAME = 'feature_to_gene_map_pam50_integrated.json'
