from configs import base_config as base

# 1. Tên định danh cho thí nghiệm
EXPERIMENT_NAME = "XGBoost_Baseline"

# 2. Tên model - PHẢI KHỚP với logic trong run_proper_cv_pipeline.py
MODEL_NAME = 'XGBoost'

# 3. Cài đặt pipeline
USE_SAMPLING = True
SAMPLING_METHOD = "BorderlineSMOTE"
MAX_LASSO_FEATURES = 2000 # Giữ tên biến để tương thích, giờ nó điều khiển LGBM
NUM_CLASSES = 4
N_SPLITS = 10 # Chạy với 10-fold CV

# Các tham số MODEL_PARAMS và TRAINING_PARAMS không cần thiết cho XGBoost
