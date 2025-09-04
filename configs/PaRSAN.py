from configs import base_config as base

# Tên thí nghiệm mới
EXPERIMENT_NAME = "PaRSAN"
MODEL_NAME = 'PaRSAN'
USE_SAMPLING = True
SAMPLING_METHOD = "BorderlineSMOTE"
NUM_CLASSES = 4
MAX_LASSO_FEATURES = 800
N_SPLITS=10

# Cấu hình kiến trúc model
MODEL_PARAMS = {
    "input_dim": None,
    "num_classes": None,
    "attention_dim": 256,
    "dropout_rate": 0.3 # Tăng cường dropout
}

# Cấu hình cho quá trình huấn luyện
TRAINING_PARAMS = {
    "optimizer": "AdamW",             # Sử dụng AdamW
    "lr": 0.00009,
    "epochs": 250,                    # Tăng epochs cho scheduler
    "batch_size": 128,
    "weight_decay": 1e-3,             # Tăng cường weight_decay
    "early_stopping_patience": 50,    # Tăng patience cho scheduler
    "loss_type": "cross_entropy",
    "label_smoothing": 0.15,           # Áp dụng Label Smoothing
    "use_scheduler": True             # Kích hoạt LR Scheduler
}

# Định nghĩa tên các file dữ liệu cho bài toán PAM50
X_TRAIN_FILENAME = 'X_train_pam50.pkl'
Y_TRAIN_FILENAME = 'y_train_pam50.pkl'
X_TEST_FILENAME = 'X_test_pam50.pkl'
Y_TEST_FILENAME = 'y_test_pam50.pkl'
PATHWAY_MAP_FILENAME = 'pathway_gene_mapping_pam50_integrated.json'
FEATURE_MAP_FILENAME = 'feature_to_gene_map_pam50_integrated.json'
