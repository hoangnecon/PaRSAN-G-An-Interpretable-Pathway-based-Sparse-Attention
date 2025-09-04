import pandas as pd
import os
import json

def load_development_data(cfg):
    """
    Tải dữ liệu development (train/validation) dựa trên tên file
    được định nghĩa trong file config.
    """
    p_path = cfg.base.GDRIVE_PROCESSED_PATH

    # Lấy tên file từ config. Nếu không có, dùng tên mặc định.
    x_train_file = getattr(cfg, 'X_TRAIN_FILENAME', cfg.base.X_TRAIN_FILENAME)
    y_train_file = getattr(cfg, 'Y_TRAIN_FILENAME', cfg.base.Y_TRAIN_FILENAME)
    pathway_file = getattr(cfg, 'PATHWAY_MAP_FILENAME', cfg.base.PATHWAY_MAP_FILENAME)
    feature_file = getattr(cfg, 'FEATURE_MAP_FILENAME', cfg.base.FEATURE_MAP_FILENAME)

    X_dev = pd.read_pickle(os.path.join(p_path, x_train_file))
    y_dev = pd.read_pickle(os.path.join(p_path, y_train_file))
    with open(os.path.join(p_path, pathway_file), 'r') as f:
        p_map = json.load(f)
    with open(os.path.join(p_path, feature_file), 'r') as f:
        f_map = json.load(f)

    print(f"-> Đã tải Development Set từ '{x_train_file}' gồm: {X_dev.shape[0]} mẫu.")
    return X_dev, y_dev, p_map, f_map

def load_test_data(cfg):
    """
    Tải dữ liệu test dựa trên tên file được định nghĩa trong file config.
    """
    p_path = cfg.base.GDRIVE_PROCESSED_PATH

    # Lấy tên file từ config. Nếu không có, dùng tên mặc định.
    x_test_file = getattr(cfg, 'X_TEST_FILENAME', cfg.base.X_TEST_FILENAME)
    y_test_file = getattr(cfg, 'Y_TEST_FILENAME', cfg.base.Y_TEST_FILENAME)

    X_test = pd.read_pickle(os.path.join(p_path, x_test_file))
    y_test = pd.read_pickle(os.path.join(p_path, y_test_file))

    print(f"-> Đã tải Final Test Set từ '{x_test_file}' gồm: {X_test.shape[0]} mẫu.")
    return X_test, y_test


def create_pam(df, p_map, f_map):
    """Tạo Ma trận Hoạt động Pathway (PAM). Giữ nguyên không đổi."""
    gene_to_feature_map = {}
    for feature, gene in f_map.items():
        if gene not in gene_to_feature_map:
            gene_to_feature_map[gene] = []
        gene_to_feature_map[gene].append(feature)

    pathway_series_list = []
    for pathway, genes in p_map.items():
        # Tạo tên an toàn cho cột, loại bỏ ký tự đặc biệt
        safe_pathway_name = pathway.replace(' ', '_').replace(':', '_').replace('/', '_')

        # Tìm các features tương ứng trong dữ liệu (df.columns)
        # Chỉ lấy các features có trong df để tính trung bình
        rna_features = [f'RNA_{f}' for g in genes if g in gene_to_feature_map for f in gene_to_feature_map.get(g, []) if f'RNA_{f}' in df.columns]
        meth_features = [f'METH_{f}' for g in genes if g in gene_to_feature_map for f in gene_to_feature_map.get(g, []) if f'METH_{f}' in df.columns]
        cnv_features = [f'CNV_{f}' for g in genes if g in gene_to_feature_map for f in gene_to_feature_map.get(g, []) if f'CNV_{f}' in df.columns]


        # Chỉ tính toán và thêm cột nếu tìm thấy features
        if rna_features:
            pathway_series_list.append(df[rna_features].mean(axis=1).rename(f'RNA_{safe_pathway_name}'))
        if meth_features:
            pathway_series_list.append(df[meth_features].mean(axis=1).rename(f'METH_{safe_pathway_name}'))
        if cnv_features:
            pathway_series_list.append(df[cnv_features].mean(axis=1).rename(f'CNV_{safe_pathway_name}'))


    if not pathway_series_list:
        return pd.DataFrame(index=df.index)

    return pd.concat(pathway_series_list, axis=1).fillna(0)
