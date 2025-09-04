import os
import pandas as pd
from sklearn.model_selection import train_test_split
from configs import base_config as cfg
import numpy as np

def main():
    print("--- BẮT ĐẦU SCRIPT 02b (Tạo Tập dữ liệu 4 LỚP) ---")
    raw_path = os.path.join(cfg.GDRIVE_PROJECT_PATH, 'data', 'raw')
    processed_path = cfg.GDRIVE_PROCESSED_PATH

    # Tải dữ liệu clinical và ID mẫu như cũ
    clinical_path = os.path.join(raw_path, "cBioportal_pancancer_clinical_data.tsv")
    clinical_df = pd.read_csv(clinical_path, sep='\t', engine='python')
    rna_cols_df = pd.read_csv(os.path.join(raw_path, "TCGA-BRCA.star_fpkm.tsv.gz"), sep='\t', engine='python', nrows=0)
    meth_cols_df = pd.read_csv(os.path.join(raw_path, "TCGA-BRCA.methylation450.tsv.gz"), sep='\t', engine='python', nrows=0)
    cnv_filename = "TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
    cnv_cols_df = pd.read_csv(os.path.join(raw_path, cnv_filename), sep='\t', engine='python', nrows=0)

    brca_df = clinical_df[clinical_df['TCGA PanCanAtlas Cancer Type Acronym'] == 'BRCA'].copy()
    pam50_df = brca_df.dropna(subset=['Subtype']).copy()

    # THAY ĐỔI QUAN TRỌNG: Loại bỏ phân nhóm 'Normal-like'
    pam50_df = pam50_df[pam50_df['Subtype'] != 'BRCA_Normal'].copy()
    print("\n-> Đã loại bỏ các mẫu thuộc phân nhóm 'Normal-like'.")

    # Mã hóa lại 4 lớp còn lại
    pam50_subtypes = sorted(pam50_df['Subtype'].unique())
    subtype_mapping = {s: i for i, s in enumerate(pam50_subtypes)}
    pam50_df['label'] = pam50_df['Subtype'].map(subtype_mapping)
    print("-> Ánh xạ các phân nhóm mới (4 lớp):")
    print(subtype_mapping)

    # Tìm các mẫu chung và chia dữ liệu như cũ
    rna_ids = set(col.rsplit('-', 1)[0] for col in rna_cols_df.columns[1:])
    meth_ids = set(col.rsplit('-', 1)[0] for col in meth_cols_df.columns[1:])
    cnv_ids = set(col.rsplit('-', 1)[0] for col in cnv_cols_df.columns[1:])
    all_samples_with_label = set(pam50_df['Patient ID'].tolist())
    common_samples = sorted(list(all_samples_with_label & rna_ids & meth_ids & cnv_ids))
    final_df = pam50_df[pam50_df['Patient ID'].isin(common_samples)].set_index('Patient ID').loc[common_samples]

    print(f"-> Số mẫu có đủ 3 omics và nhãn (4 lớp): {len(final_df)}")

    X_train_ids, X_test_ids, y_train, y_test = train_test_split(
        final_df.index, final_df['label'], test_size=0.2,
        random_state=cfg.RANDOM_STATE, stratify=final_df['label']
    )

    print(f"-> Chia {len(final_df)} mẫu thành {len(X_train_ids)} train và {len(X_test_ids)} test.")

    # Ghi đè lên các file cũ để pipeline tiếp theo có thể chạy mà không cần sửa đổi
    pd.Series(X_train_ids).to_csv(os.path.join(processed_path, 'train_ids_pam50.txt'), index=False, header=False)
    pd.Series(X_test_ids).to_csv(os.path.join(processed_path, 'test_ids_pam50.txt'), index=False, header=False)
    y_train.to_pickle(os.path.join(processed_path, 'y_train_pam50.pkl'))
    y_test.to_pickle(os.path.join(processed_path, 'y_test_pam50.pkl'))

    print("\nPhân bố lớp trong tập train mới:")
    print(y_train.value_counts().sort_index())
    print("\n--- KẾT THÚC SCRIPT ---")

if __name__ == "__main__":
    main()
