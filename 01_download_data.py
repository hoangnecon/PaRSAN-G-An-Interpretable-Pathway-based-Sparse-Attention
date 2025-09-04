import os
import requests
from tqdm import tqdm
import configs.base_config as config
import shutil

def main():
    print("--- BẮT ĐẦU SCRIPT 01 (Tải Dữ liệu multi-omics và PAM50) ---")
    data_urls = {
        "RNA_seq": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_fpkm.tsv.gz",
        "Methylation": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz",
        "Copy_Number": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
        "Clinical_TCGA": "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz",
        "Clinical_Pancancer": "https://raw.githubusercontent.com/GerkeLab/TCGAclinical/master/data/cBioportal_data.tsv"
    }
    output_dir = os.path.join(config.GDRIVE_PROJECT_PATH, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    for data_type, url in data_urls.items():
        local_filename = url.split('/')[-1]
        local_filepath = os.path.join(output_dir, local_filename)

        # Đổi tên file pancancer cho dễ nhớ
        if 'cBioportal_data.tsv' in local_filename:
            local_filepath = os.path.join(output_dir, "cBioportal_pancancer_clinical_data.tsv")

        if os.path.exists(local_filepath):
            print(f"File {local_filepath} đã tồn tại. Bỏ qua tải xuống.")
            continue

        try:
            print(f"-> Đang tải {data_type} từ {url}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(local_filepath, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=local_filename
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"-> Tải thành công!")
        except Exception as e:
            print(f"Lỗi khi tải {url}: {e}")
    print("--- KẾT THÚC SCRIPT 01 ---")

if __name__ == '__main__':
    main()
