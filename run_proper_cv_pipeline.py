import argparse
import importlib
import os
import joblib
import pandas as pd
import numpy as np
import torch
import json
import random
import time

from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src import data_utils, models, training_utils


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"-> Seed toàn cục đã được thiết lập thành: {seed}")


def main(config_name):
    try:
        config = importlib.import_module(f"configs.{config_name}")
        print(f"--- BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ K-FOLD CV: {config.EXPERIMENT_NAME} ---")
    except ImportError:
        print(f"LỖI: Không tìm thấy file cấu hình 'configs/{config_name}.py'")
        return

    print("\n" + "="*70)
    print("--- CẤU HÌNH THÍ NGHIỆM ---"); print(f"Thí nghiệm: {config.EXPERIMENT_NAME}"); print(f"Mô hình: {config.MODEL_NAME}"); print(f"Số Lớp: {config.NUM_CLASSES}")
    use_sampling = getattr(config, 'USE_SAMPLING', False)
    sampling_method = getattr(config, 'SAMPLING_METHOD', 'None')
    print(f"Sử dụng cân bằng dữ liệu: {use_sampling} ({sampling_method})")
    print("="*70)

    seed_everything(config.base.RANDOM_STATE)

    print("\n[Bước 1] Tải và gộp toàn bộ dữ liệu...")
    try:
        p_path = config.base.GDRIVE_PROCESSED_PATH
        X_train_part = pd.read_pickle(os.path.join(p_path, 'X_train_pam50.pkl'))
        y_train_part = pd.read_pickle(os.path.join(p_path, 'y_train_pam50.pkl'))
        X_test_part = pd.read_pickle(os.path.join(p_path, 'X_test_pam50.pkl'))
        y_test_part = pd.read_pickle(os.path.join(p_path, 'y_test_pam50.pkl'))
        X_full = pd.concat([X_train_part, X_test_part]); y_full = pd.concat([y_train_part, y_test_part])
        with open(os.path.join(p_path, 'pathway_gene_mapping_pam50_integrated.json'), 'r') as f: pathway_map = json.load(f)
        with open(os.path.join(p_path, 'feature_to_gene_map_pam50_integrated.json'), 'r') as f: feature_map = json.load(f)
        print(f"-> Đã tải thành công và gộp thành bộ dữ liệu đầy đủ gồm {len(X_full)} mẫu.")
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file dữ liệu. Lỗi: {e}"); return

    n_splits = config.N_SPLITS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.base.RANDOM_STATE)

    report_dir = os.path.join(config.base.REPORT_SAVE_PATH, config.EXPERIMENT_NAME)
    os.makedirs(report_dir, exist_ok=True)

    # <<< THÊM MỚI: Tạo thư mục để lưu các model của từng fold >>>
    models_dir = os.path.join(config.base.MODEL_SAVE_PATH, config.EXPERIMENT_NAME)
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n[Bước 2] Bắt đầu {n_splits}-Fold Cross-Validation trên toàn bộ dữ liệu...")
    print(f"-> Checkpoint kết quả của từng fold sẽ được lưu tại: {report_dir}")
    print(f"-> Model và các đối tượng pipeline của từng fold sẽ được lưu tại: {models_dir}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
        current_fold_num = fold + 1
        fold_start_time = time.time()

        result_path = os.path.join(report_dir, f'fold_{current_fold_num}_results.json')
        if os.path.exists(result_path):
            print(f"\n--- CHECKPOINT: Đã tìm thấy kết quả cho Fold {current_fold_num}. Bỏ qua... ---")
            continue

        print(f"\n--- Đang xử lý Fold {current_fold_num}/{n_splits} ---")

        # <<< THÊM MỚI: Tạo thư mục con cho fold hiện tại >>>
        fold_model_dir = os.path.join(models_dir, f"fold_{current_fold_num}")
        os.makedirs(fold_model_dir, exist_ok=True)

        X_train_fold, y_train_fold = X_full.iloc[train_idx], y_full.iloc[train_idx]
        X_test_fold, y_test_fold = X_full.iloc[test_idx], y_full.iloc[test_idx]

        if use_sampling:
            print(f"    -> (a) Áp dụng {sampling_method}...")
            if sampling_method == 'BorderlineSMOTE': sampler = BorderlineSMOTE(random_state=config.base.RANDOM_STATE, k_neighbors=5)
            elif sampling_method == 'SMOTE': sampler = SMOTE(random_state=config.base.RANDOM_STATE)
            else: sampler = None
            if sampler: X_train_fold, y_train_fold = sampler.fit_resample(X_train_fold, y_train_fold)

        print("    -> (b) Đang tạo ma trận PAM...")
        pam_train = data_utils.create_pam(X_train_fold, pathway_map, feature_map)
        pam_test = data_utils.create_pam(X_test_fold, pathway_map, feature_map)
        pam_test = pam_test.reindex(columns=pam_train.columns, fill_value=0)

        print("    -> (c) Đang chuẩn hóa dữ liệu...")
        scaler = StandardScaler()
        pam_train_scaled = scaler.fit_transform(pam_train); pam_test_scaled = scaler.transform(pam_test)

        max_lasso_features = getattr(config, 'MAX_LASSO_FEATURES', 2000)
        print(f"    -> (d) Lựa chọn đặc trưng bằng LassoCV (tối đa {max_lasso_features})...")
        lasso_cv = LassoCV(cv=3, random_state=config.base.RANDOM_STATE, n_jobs=-1, max_iter=2000)
        selector = SelectFromModel(lasso_cv, threshold=-np.inf, max_features=max_lasso_features).fit(pam_train_scaled, y_train_fold)
        X_train_selected = selector.transform(pam_train_scaled); X_test_selected = selector.transform(pam_test_scaled)
        print(f"       -> Đã chọn {X_train_selected.shape[1]} đặc trưng.")

        print(f"    -> (e) Huấn luyện mô hình {config.MODEL_NAME}...")
        model_name = config.MODEL_NAME.lower()
        history = None

        if model_name.startswith('parsan'):
            model_params = config.MODEL_PARAMS.copy(); model_params['input_dim'] = X_train_selected.shape[1]; model_params['num_classes'] = config.NUM_CLASSES
            if config.MODEL_NAME == 'PaRSAN': model = models.PaRSAN(**model_params)
            else: model = models.PaRSAN_G(**model_params)
            model, history = training_utils.train_pytorch_model(model, X_train_selected, y_train_fold, X_test_selected, y_test_fold, config)
        else:
            if model_name == 'logisticregression': model = LogisticRegression(random_state=config.base.RANDOM_STATE, max_iter=1000, class_weight='balanced', n_jobs=-1)
            elif model_name == 'xgboost': model = XGBClassifier(random_state=config.base.RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
            else: raise ValueError(f"Model {config.MODEL_NAME} không được hỗ trợ.")
            model.fit(X_train_selected, y_train_fold)

        # <<< THÊM MỚI: Lưu scaler, selector và model của fold hiện tại >>>
        joblib.dump(scaler, os.path.join(fold_model_dir, "scaler.joblib"))
        joblib.dump(selector, os.path.join(fold_model_dir, "selector.joblib"))
        if model_name.startswith('parsan'):
            torch.save(model.state_dict(), os.path.join(fold_model_dir, "model.pth"))
        else:
            joblib.dump(model, os.path.join(fold_model_dir, "model.joblib"))
        print(f"    -> Đã lưu model, scaler, selector của Fold {current_fold_num} vào {fold_model_dir}")

        print(f"    -> (f) Đánh giá trên tập test của fold {current_fold_num}...")
        # ... (Phần đánh giá giữ nguyên)
        if model_name.startswith('parsan'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); model.eval()
            test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_test_logits = model(test_tensor); y_test_pred_proba = torch.softmax(y_test_logits, dim=1).cpu().numpy(); y_test_pred = torch.argmax(y_test_logits, dim=1).cpu().numpy()
        else:
            y_test_pred_proba = model.predict_proba(X_test_selected); y_test_pred = model.predict(X_test_selected)

        fold_time = time.time() - fold_start_time
        result_data = {
            'fold': current_fold_num, 'auc': roc_auc_score(y_test_fold, y_test_pred_proba, multi_class='ovr', average='macro'),
            'accuracy': accuracy_score(y_test_fold, y_test_pred), 'f1_macro': f1_score(y_test_fold, y_test_pred, average='macro'),
            'time_seconds': fold_time, 'training_history': history
        }
        print(f"  -> Kết quả Fold {current_fold_num}: AUC={result_data['auc']:.4f}, Accuracy={result_data['accuracy']:.4f}, F1={result_data['f1_macro']:.4f} (Thời gian: {fold_time:.2f}s)")

        with open(result_path, 'w') as f: json.dump(result_data, f, indent=4)
        print(f"  -> Đã lưu checkpoint kết quả tại: {result_path}")

        del model, scaler, selector; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- 4. Báo cáo Kết quả Cuối cùng ---
    print("\n\n" + "="*70)
    print(f"--- KẾT QUẢ CUỐI CÙNG ({n_splits}-FOLD CROSS-VALIDATION) ---"); print(f"Thí nghiệm: {config.EXPERIMENT_NAME}"); print("="*70)

    all_fold_results = []
    for i in range(1, n_splits + 1):
        result_path = os.path.join(report_dir, f'fold_{i}_results.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f: all_fold_results.append(json.load(f))

    if not all_fold_results: print("LỖI: Không tìm thấy file kết quả nào để tổng hợp."); return

    results_df = pd.DataFrame(all_fold_results).drop(columns=['training_history'], errors='ignore')
    print("Chi tiết kết quả từng fold:"); print(results_df.to_string(index=False))
    print("\n--- Trung bình và Độ lệch chuẩn ---")
    print(f"Macro-AUC:      {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"Accuracy:       {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Macro-F1 Score: {results_df['f1_macro'].mean():.4f} ± {results_df['f1_macro'].std():.4f}")

    results_df.to_csv(os.path.join(report_dir, 'kfold_cv_results_summary.csv'), index=False)
    print(f"\n-> Đã lưu bảng tổng hợp kết quả K-Fold CV vào: {report_dir}")
    print("\n--- KẾT THÚC QUY TRÌNH ĐÁNH GIÁ ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy quy trình đánh giá K-Fold CV chuẩn SOTA với Checkpoint.")
    parser.add_argument('--config', type=str, required=True, help="Tên file cấu hình của thí nghiệm.")
    args = parser.parse_args()
    main(args.config)
