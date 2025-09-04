from configs import base_config as base

# Tên thí nghiệm mới, rõ ràng hơn
EXPERIMENT_NAME = "PaRSAN_G"
MODEL_NAME = 'PaRSAN_G'
NUM_CLASSES = 4
MAX_LASSO_FEATURES = 800 # Giữ nguyên số features từ thí nghiệm tốt nhất
N_SPLITS=10

# --- Cài đặt cho việc cân bằng dữ liệu ---
USE_SAMPLING = True             # Bật/Tắt chức năng cân bằng dữ liệu
SAMPLING_METHOD = "BorderlineSMOTE" # Có thể đổi thành "SMOTE" hoặc các phương pháp khác trong tương lai

# --- Các tham số khác giữ nguyên từ file config tốt nhất của bạn ---

# Cấu hình kiến trúc model
MODEL_PARAMS = {
    "input_dim": None,
    "num_classes": None,
    "attention_hidden_layers": [256, 128],
    "gate_hidden_layers": [256, 128],
    "classifier_hidden_layers": [128, 64],
    "dropout_rate": 0.40
}

# Cấu hình cho quá trình huấn luyện
TRAINING_PARAMS = {
    "optimizer": "AdamW",
    "lr": 5e-05,
    "epochs": 250,
    "batch_size": 128,
    "weight_decay": 0.01,
    "early_stopping_patience": 50,
    "loss_type": "cross_entropy",
    "label_smoothing": 0.15,
    "use_scheduler": True
}

# Định nghĩa tên các file dữ liệu cho bài toán PAM50
X_TRAIN_FILENAME = 'X_train_pam50.pkl'
Y_TRAIN_FILENAME = 'y_train_pam50.pkl'
X_TEST_FILENAME = 'X_test_pam50.pkl'
Y_TEST_FILENAME = 'y_test_pam50.pkl'
PATHWAY_MAP_FILENAME = 'pathway_gene_mapping_pam50_integrated.json'
FEATURE_MAP_FILENAME = 'feature_to_gene_map_pam50_integrated.json'
