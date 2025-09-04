import os
import pandas as pd
import numpy as np
from configs import base_config as cfg
import gc
import json
import gzip
import mygene
from tqdm import tqdm
from sklearn.feature_selection import f_classif

def get_rna_cnv_mappable_features(raw_path, omics_files, processed_path):
    """
    Hàm này quét qua các file RNA và CNV để tìm features có thể ánh xạ.
    (Nội dung hàm này không thay đổi)
    """
    feature_map_full_path = os.path.join(processed_path, 'rna_cnv_mappable_features.json')

    if os.path.exists(feature_map_full_path):
        print("-> CHECKPOINT: Tìm thấy file ánh xạ RNA/CNV đầy đủ. Đang tải...")
        with open(feature_map_full_path, 'r') as f:
            return json.load(f)

    print("-> Bước 1: Tổng hợp và ánh xạ features RNA/CNV...")
    temp_feature_file = os.path.join(processed_path, "rna_cnv_features_temp.txt")

    rna_cnv_files = [f for f in omics_files if 'methylation' not in f]

    if os.path.exists(temp_feature_file): os.remove(temp_feature_file)
    for file_name in rna_cnv_files:
        file_path = os.path.join(raw_path, file_name)
        if not os.path.exists(file_path): continue
        with gzip.open(file_path, 'rt', errors='ignore') as f_in, open(temp_feature_file, 'a', encoding='utf-8') as f_out:
            next(f_in)
            for line in f_in:
                if line and '\t' in line:
                    feature_id = line.split('\t', 1)[0].strip()
                    if feature_id: f_out.write(f"{feature_id}\n")

    with open(temp_feature_file, 'r', encoding='utf-8') as f:
        all_features = sorted(list(set([line.strip() for line in f if line.strip()])))
    os.remove(temp_feature_file)

    cleaned_id_map = {f.split('.')[0] if f.startswith('ENSG') else f: f for f in all_features}
    query_list = list(cleaned_id_map.keys())
    mg = mygene.MyGeneInfo()
    feature_to_gene_map = {}
    batch_size = 100000
    print(f"-> Bắt đầu truy vấn MyGene cho {len(query_list)} IDs...")
    for i in tqdm(range(0, len(query_list), batch_size), desc="MyGene Queries"):
        batch = query_list[i:i + batch_size]
        try:
            gene_info_batch = mg.querymany(batch, scopes='all', fields='symbol', species='human', returnall=True)
            cleaned_to_symbol_map_batch = {res['query']: res['symbol'] for res in gene_info_batch['out'] if 'symbol' in res}
            for cleaned_id, original_id in cleaned_id_map.items():
                if cleaned_id in cleaned_to_symbol_map_batch:
                    feature_to_gene_map[original_id] = cleaned_to_symbol_map_batch[cleaned_id]
        except Exception as e: print(f"Lỗi ở batch {i//batch_size + 1}: {e}")

    with open(feature_map_full_path, 'w') as f:
        json.dump(feature_to_gene_map, f)

    return feature_to_gene_map

def process_omic_data_optimized(file_name, data_prefix, train_ids, test_ids, y_train_labels, top_n_features, raw_path, feature_to_gene_map, p_value_threshold=0.05, chunk_size=50000):
    """
    Hàm xử lý file omics với phương pháp chọn lọc feature 2 bước:
    1. Lọc theo P-value (ANOVA F-test) để tìm các features liên quan đến các subtype.
    2. Từ các features đã lọc, chọn top N có phương sai cao nhất.
    """
    print(f"\n--- Đang xử lý file: {file_name} ---")
    file_path = os.path.join(raw_path, file_name)

    valid_features = set()
    if "star_fpkm" in file_name or "Gistic2_CopyNumber" in file_name:
        valid_features = set(feature_to_gene_map.keys())
    elif "methylation" in file_name:
        print("-> Đọc tất cả probes từ file methylation...")
        chunks = pd.read_csv(file_path, sep='\t', engine='python', chunksize=1000, index_col=0)
        for chunk in chunks:
            valid_features.update([f for f in chunk.index if f.startswith('cg')])
        del chunks
        gc.collect()

    if not valid_features:
        print(f"!!! CẢNH BÁO: Không tìm thấy features hợp lệ. Bỏ qua file {file_name}.")
        return pd.DataFrame(index=pd.Index([], name='sample')), pd.DataFrame(index=pd.Index([], name='sample')), []

    print("-> Giai đoạn 1: Tải dữ liệu training và chọn lọc features...")
    train_data_chunks = []
    chunks_iterator = pd.read_csv(file_path, sep='\t', engine='python', chunksize=chunk_size, index_col=0)
    for chunk in tqdm(chunks_iterator, desc=f"   -> Đọc chunk {file_name}"):
        chunk.columns = chunk.columns.str.slice(0, 12)
        chunk = chunk.loc[:, ~chunk.columns.duplicated(keep='first')]

        chunk_filtered = chunk.loc[chunk.index.intersection(valid_features)]
        train_cols_in_chunk = [tid for tid in train_ids if tid in chunk_filtered.columns]

        if train_cols_in_chunk:
            train_data_chunks.append(chunk_filtered[train_cols_in_chunk])
        del chunk, chunk_filtered
        gc.collect()

    if not train_data_chunks:
        print(f"!!! CẢNH BÁO: Không tìm thấy dữ liệu training trong file {file_name}.")
        return pd.DataFrame(index=pd.Index([], name='sample')), pd.DataFrame(index=pd.Index([], name='sample')), []

    # Ghép các chunk lại
    train_df_full_temp = pd.concat(train_data_chunks, axis=1)
    del train_data_chunks
    gc.collect()

    # SỬA LỖI: Loại bỏ các cột bị trùng lặp sau khi đã ghép các chunk
    train_df_full_temp = train_df_full_temp.loc[:, ~train_df_full_temp.columns.duplicated(keep='first')]

    # Chuyển vị (samples là hàng, features là cột)
    train_df_full = train_df_full_temp.T
    del train_df_full_temp

    # Đảm bảo thứ tự của dữ liệu và nhãn khớp nhau
    train_df_full = train_df_full.loc[y_train_labels.index]

    print(f"-> Thực hiện kiểm định ANOVA để lọc features (p < {p_value_threshold})...")
    train_df_full.fillna(0, inplace=True)
    _, p_values = f_classif(train_df_full, y_train_labels)

    p_values_series = pd.Series(p_values, index=train_df_full.columns)
    significant_features = p_values_series[p_values_series < p_value_threshold].index

    print(f"-> Tìm thấy {len(significant_features)} features có ý nghĩa thống kê.")

    if len(significant_features) == 0:
        print(f"!!! CẢNH BÁO: Không có feature nào vượt qua ngưỡng p-value. Bỏ qua file {file_name}.")
        return pd.DataFrame(index=pd.Index([], name='sample')), pd.DataFrame(index=pd.Index([], name='sample')), []

    print(f"-> Chọn top {top_n_features} features có phương sai cao nhất từ tập đã lọc...")
    variances = train_df_full[significant_features].var()
    num_to_select = min(top_n_features, len(significant_features))
    top_features = variances.nlargest(num_to_select).index.tolist()

    print(f"-> Đã chọn top {len(top_features)} features cuối cùng.")
    del train_df_full, variances, p_values_series
    gc.collect()

    print("-> Giai đoạn 2: Trích xuất dữ liệu cho các features đã chọn...")
    final_df_rows = []
    chunks_iterator_2 = pd.read_csv(file_path, sep='\t', engine='python', chunksize=chunk_size, index_col=0)
    for chunk in tqdm(chunks_iterator_2, desc=f"   -> Trích xuất chunk {file_name}"):
        chunk.columns = chunk.columns.str.slice(0, 12)
        chunk = chunk.loc[:, ~chunk.columns.duplicated(keep='first')]
        relevant_rows = chunk.loc[chunk.index.intersection(top_features)]
        if not relevant_rows.empty:
            final_df_rows.append(relevant_rows)

    if not final_df_rows:
        print(f"!!! CẢNH BÁO: Không tìm thấy dữ liệu cho top features trong file {file_name}")
        return pd.DataFrame(index=pd.Index([], name='sample')), pd.DataFrame(index=pd.Index([], name='sample')), []

    df_final = pd.concat(final_df_rows).loc[top_features]
    del final_df_rows
    gc.collect()

    X_train_omic = df_final.reindex(columns=train_ids).T.fillna(0)
    X_test_omic = df_final.reindex(columns=test_ids).T.fillna(0)

    X_train_omic.columns = [f'{data_prefix}_{c}' for c in top_features]
    X_test_omic.columns = X_train_omic.columns

    return X_train_omic, X_test_omic, top_features

def main():
    print("--- BẮT ĐẦU SCRIPT 03 (Tạo Feature Sets cho Phân nhóm PAM50")
    processed_path = cfg.GDRIVE_PROCESSED_PATH
    raw_path = os.path.join(cfg.GDRIVE_PROJECT_PATH, 'data', 'raw')

    try:
        train_ids = pd.read_csv(os.path.join(processed_path, 'train_ids_pam50.txt'), header=None)[0].tolist()
        test_ids = pd.read_csv(os.path.join(processed_path, 'test_ids_pam50.txt'), header=None)[0].tolist()
        y_train = pd.read_pickle(os.path.join(processed_path, 'y_train_pam50.pkl'))
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file ID hoặc file nhãn. Vui lòng chạy script 02 trước. Lỗi: {e}")
        return

    omics_files = ["TCGA-BRCA.star_fpkm.tsv.gz", "TCGA-BRCA.methylation450.tsv.gz", "TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"]
    feature_to_gene_map = get_rna_cnv_mappable_features(raw_path, omics_files, processed_path)
    if not feature_to_gene_map:
        print("LỖI: Không thể tạo Feature-Gene Map. Dừng script.")
        return

    X_train_rna, X_test_rna, rna_features = process_omic_data_optimized(omics_files[0], "RNA", train_ids, test_ids, y_train, cfg.TOP_N_FEATURES, raw_path, feature_to_gene_map)
    X_train_meth, X_test_meth, meth_features = process_omic_data_optimized(omics_files[1], "METH", train_ids, test_ids, y_train, cfg.TOP_N_FEATURES, raw_path, feature_to_gene_map)
    X_train_cnv, X_test_cnv, cnv_features = process_omic_data_optimized(omics_files[2], "CNV", train_ids, test_ids, y_train, cfg.TOP_N_FEATURES, raw_path, feature_to_gene_map)

    print("\n--- Đang ghép nối các ma trận ---")
    X_train = pd.concat([X_train_rna, X_train_meth, X_train_cnv], axis=1)
    X_test = pd.concat([X_test_rna, X_test_meth, X_test_cnv], axis=1)

    if X_train.empty or X_test.empty:
        print("LỖI: Ma trận dữ liệu rỗng sau khi ghép nối. Dừng script.")
        return

    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    X_train.to_pickle(os.path.join(processed_path, 'X_train_pam50.pkl'))
    X_test.to_pickle(os.path.join(processed_path, 'X_test_pam50.pkl'))

    selected_features_all = rna_features + meth_features + cnv_features
    final_feature_to_gene_map = {f: feature_to_gene_map[f] for f in selected_features_all if f in feature_to_gene_map}

    with open(os.path.join(processed_path, 'feature_to_gene_map_pam50.json'), 'w') as f:
        json.dump(final_feature_to_gene_map, f)

    print(f"\n-> Kích thước X_train cuối cùng: {X_train.shape}")
    print(f"-> Đã lưu file feature mới (X_train_pam50.pkl, X_test_pam50.pkl) và ánh xạ gene vào: {processed_path}")
    print("--- KẾT THÚC SCRIPT ---")

if __name__ == "__main__":
    main()
