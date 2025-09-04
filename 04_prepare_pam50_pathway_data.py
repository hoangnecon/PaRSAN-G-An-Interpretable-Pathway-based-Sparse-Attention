import os
import pandas as pd
import json
import gseapy as gp
import requests
from tqdm import tqdm
from configs import base_config as cfg
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def download_file(url, output_path):
    """Tải file từ URL với thanh tiến trình."""
    if os.path.exists(output_path):
        print(f"-> File chú giải đã tồn tại tại: {output_path}. Bỏ qua tải xuống.")
        return
    print(f"-> Đang tải file chú giải từ {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(output_path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("-> Tải thành công!")
    except Exception as e:
        print(f"LỖI khi tải file chú giải: {e}")
        raise

def create_methylation_to_gene_map(annotation_file_path):
    """Tạo bản đồ ánh xạ từ probe ID sang gene symbol."""
    print("-> Đang tạo bản đồ ánh xạ cho Methylation Probes...")
    try:
        # Đọc file CSV, bỏ qua các dòng comment ở đầu nếu có
        df = pd.read_csv(
            annotation_file_path,
            usecols=['IlmnID', 'UCSC_RefGene_Name'],
            dtype={'IlmnID': str, 'UCSC_RefGene_Name': str},
            skiprows=7  # Bỏ qua 7 dòng header của file manifest
        )
        df.dropna(subset=['UCSC_RefGene_Name'], inplace=True)
        df = df.set_index('IlmnID')
        # Chỉ lấy gene đầu tiên nếu có nhiều gene được liệt kê (ví dụ: 'TP53;KRAS')
        probe_map = df['UCSC_RefGene_Name'].apply(lambda x: x.split(';')[0]).to_dict()
        print(f"-> Đã tạo bản đồ cho {len(probe_map)} probes.")
        return probe_map
    except Exception as e:
        print(f"LỖI khi xử lý file chú giải: {e}")
        return {}

def main():
    print("--- BẮT ĐẦU SCRIPT 04")
    processed_path = cfg.GDRIVE_PROCESSED_PATH
    raw_path = os.path.join(cfg.GDRIVE_PROJECT_PATH, 'data', 'raw')

    # --- Bước 1: Xử lý ánh xạ cho Methylation ---
    annotation_url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
    annotation_filename = os.path.basename(annotation_url)
    annotation_filepath = os.path.join(raw_path, annotation_filename)

    download_file(annotation_url, annotation_filepath)
    probe_to_gene_map = create_methylation_to_gene_map(annotation_filepath)

    # --- Bước 2: Tải các bản đồ và dữ liệu hiện có ---
    feature_to_gene_map_path = os.path.join(processed_path, 'feature_to_gene_map_pam50.json')
    try:
        with open(feature_to_gene_map_path, 'r') as f:
            rna_cnv_to_gene_map = json.load(f)
        X_train = pd.read_pickle(os.path.join(processed_path, 'X_train_pam50.pkl'))
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file cần thiết: {e}. Vui lòng chạy lại script 03.")
        return

    # --- Bước 3: Tích hợp các bản đồ ánh xạ ---
    print("\n-> Đang tích hợp các bản đồ ánh xạ (RNA, CNV, METH)...")
    integrated_feature_map = rna_cnv_to_gene_map.copy()

    # Thêm các ánh xạ từ methylation
    methylation_features_in_model = [col for col in X_train.columns if col.startswith('METH_')]
    meth_probes = [col.split('_', 1)[1] for col in methylation_features_in_model]

    mapped_meth_count = 0
    for probe in meth_probes:
        if probe in probe_to_gene_map:
            integrated_feature_map[probe] = probe_to_gene_map[probe]
            mapped_meth_count += 1

    print(f"-> Ánh xạ thành công {mapped_meth_count} / {len(meth_probes)} features methylation.")
    print(f"-> Tổng số features được ánh xạ (tích hợp): {len(integrated_feature_map)}")

    # Lưu bản đồ tích hợp để sử dụng sau này
    integrated_map_path = os.path.join(processed_path, 'feature_to_gene_map_pam50_integrated.json')
    with open(integrated_map_path, 'w') as f:
        json.dump(integrated_feature_map, f)
    print(f"-> Đã lưu bản đồ ánh xạ tích hợp tại: {integrated_map_path}")

    # --- Bước 4: Tạo Pathway Map từ danh sách gene TÍCH HỢP ---
    pathway_map_path = os.path.join(processed_path, 'pathway_gene_mapping_pam50_integrated.json')
    if os.path.exists(pathway_map_path):
        print(f"\n-> CHECKPOINT: Tìm thấy file pathway map TÍCH HỢP tại '{pathway_map_path}'. Bỏ qua tạo mới.")
        return

    print("\n-> Chuẩn bị tạo pathway map từ danh sách gene tích hợp...")
    all_features_in_model = [col.split('_', 1)[1] for col in X_train.columns]

    # Lấy danh sách gene đầy đủ từ bản đồ tích hợp
    gene_list = [integrated_feature_map.get(f) for f in all_features_in_model]
    unique_genes_in_model = sorted(list(set(filter(None, gene_list))))

    print(f"-> Chuẩn bị truy vấn Enrichr cho {len(unique_genes_in_model)} genes từ cả 3 lớp omics...")

    if not unique_genes_in_model:
        pathway_gene_mapping = {}
    else:
        # Sử dụng nhiều thư viện pathway để có kết quả toàn diện hơn
        gene_sets = ['KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023']
        enr = gp.enrichr(gene_list=unique_genes_in_model, gene_sets=gene_sets, organism='Human', outdir=None)
        pathway_gene_mapping = {row['Term']: row['Genes'].split(';') for _, row in enr.results.iterrows()}

    with open(pathway_map_path, 'w') as f:
        json.dump(pathway_gene_mapping, f, indent=4)

    print(f"-> Đã lưu file ánh xạ pathway TÍCH HỢP mới cho {len(pathway_gene_mapping)} pathways.")
    print("--- KẾT THÚC SCRIPT ---")

if __name__ == '__main__':
    main()
