import os
import pandas as pd
from collections import defaultdict

def count_images_by_class(base_dir):
    """
    각 클래스별 이미지 개수를 계산합니다.
    
    Args:
        base_dir: 데이터셋의 기본 디렉토리 경로
        
    Returns:
        결과를 담은 DataFrame
    """
    results = defaultdict(dict)
    
    # EO 이미지 카운트
    eo_dir = os.path.join(base_dir, "EO_Train")
    for class_name in os.listdir(eo_dir):
        class_path = os.path.join(eo_dir, class_name)
        if os.path.isdir(class_path):
            # 이미지 파일만 카운트 (확장자 체크)
            image_count = len([f for f in os.listdir(class_path) 
                               if os.path.isfile(os.path.join(class_path, f)) and 
                               f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
            results[class_name]['EO'] = image_count
    
    # SAR 이미지 카운트
    sar_dir = os.path.join(base_dir, "SAR_Train")
    for class_name in os.listdir(sar_dir):
        class_path = os.path.join(sar_dir, class_name)
        if os.path.isdir(class_path):
            # 이미지 파일만 카운트 (확장자 체크)
            image_count = len([f for f in os.listdir(class_path) 
                               if os.path.isfile(os.path.join(class_path, f)) and 
                               f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
            results[class_name]['SAR'] = image_count
    
    # DataFrame으로 변환
    df = pd.DataFrame.from_dict(results, orient='index')
    df['Total'] = df['EO'] + df['SAR']
    df = df.sort_values('Total', ascending=False)
    
    return df

# 트레이닝 데이터셋 카운트
print("===== 트레이닝 데이터셋 =====")
train_counts = count_images_by_class("/home/whisper2024/PBVS/resampled/train")
print(train_counts)
print(f"총 EO 이미지: {train_counts['EO'].sum()}")
print(f"총 SAR 이미지: {train_counts['SAR'].sum()}")
print(f"총 이미지: {train_counts['Total'].sum()}")
print("\n")

# 검증 데이터셋 카운트
print("===== 검증 데이터셋 =====")
val_counts = count_images_by_class("/home/whisper2024/PBVS/resampled/val")
print(val_counts)
print(f"총 EO 이미지: {val_counts['EO'].sum()}")
print(f"총 SAR 이미지: {val_counts['SAR'].sum()}")
print(f"총 이미지: {val_counts['Total'].sum()}")