import os

# Đường dẫn chung của toàn bộ dự án
GDRIVE_PROJECT_PATH = '/content/drive/MyDrive/NCKH_Breast_Cancer'
# Đường dẫn đến thư mục chứa dữ liệu đã qua xử lý
GDRIVE_PROCESSED_PATH = os.path.join(GDRIVE_PROJECT_PATH, 'data', 'processed')
# Đường dẫn để lưu các model đã huấn luyện
MODEL_SAVE_PATH = os.path.join(GDRIVE_PROJECT_PATH, 'models')
# Đường dẫn để lưu các báo cáo, biểu đồ
REPORT_SAVE_PATH = os.path.join(GDRIVE_PROJECT_PATH, 'reports')

# --- Tên file MẶC ĐỊNH cho bài toán Tumor/Normal ---
# Đây là tên file mặc định, các script cụ thể có thể ghi đè
TRAIN_IDS_FILENAME = 'train_ids.txt'
TEST_IDS_FILENAME = 'test_ids.txt'

X_TRAIN_FILENAME = 'X_train.pkl'
Y_TRAIN_FILENAME = 'y_train.pkl'
X_TEST_FILENAME = 'X_test.pkl'
Y_TEST_FILENAME = 'y_test.pkl'

PATHWAY_MAP_FILENAME = 'pathway_gene_mapping.json'
FEATURE_MAP_FILENAME = 'feature_to_gene_map.json'

# --- Các tham số chung dùng cho nhiều thí nghiệm ---
RANDOM_STATE = 42
N_SPLITS_CV = 5
TOP_N_FEATURES = 3000
