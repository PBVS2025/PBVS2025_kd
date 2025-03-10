import os
import shutil
import glob

def copy_matching_eo_images():
    """
    SAR 데이터셋의 각 이미지와 동일한 파일명을 가진 EO 이미지만 복사합니다.
    """
    # 소스 및 대상 경로 설정
    sar_base_dir = "/home/whisper2024/PBVS/Unicorn_Dataset/SAR_Train_Filtered"
    eo_source_dir = "/home/whisper2024/PBVS/Unicorn_Dataset/EO_Train"
    eo_target_dir = "/home/whisper2024/PBVS/Unicorn_Dataset/EO_Train_Filtered"
    
    # EO_Train 대상 디렉토리가 없으면 생성
    if not os.path.exists(eo_target_dir):
        os.makedirs(eo_target_dir)
    
    # 각 클래스 처리
    classes = os.listdir(sar_base_dir)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(sar_base_dir, cls)) and cls != "EO_Train"]
    
    print(f"처리할 클래스: {classes}")
    total_matched = 0
    total_missing = 0
    
    for class_name in classes:
        print(f"\n클래스 '{class_name}' 처리 중...")
        
        # 소스와 대상 경로
        sar_class_dir = os.path.join(sar_base_dir, class_name)
        eo_source_class_dir = os.path.join(eo_source_dir, class_name)
        eo_target_class_dir = os.path.join(eo_target_dir, class_name)
        
        # 대상 디렉토리가 없으면 생성
        if not os.path.exists(eo_target_class_dir):
            os.makedirs(eo_target_class_dir)
        
        # SAR 이미지 파일명 가져오기
        sar_images = glob.glob(os.path.join(sar_class_dir, "*"))
        sar_filenames = [os.path.basename(img_path) for img_path in sar_images]
        
        print(f"'{class_name}' 클래스의 SAR 이미지 수: {len(sar_filenames)}")
        
        matched_count = 0
        missing_count = 0
        missing_files = []
        
        # 각 SAR 이미지에 대응하는 EO 이미지 찾기 및 복사
        for sar_filename in sar_filenames:
            eo_source_path = os.path.join(eo_source_class_dir, sar_filename)
            
            # 동일한 이름의 EO 이미지가 있는지 확인하고 복사
            if os.path.exists(eo_source_path):
                eo_target_path = os.path.join(eo_target_class_dir, sar_filename)
                shutil.copy2(eo_source_path, eo_target_path)
                matched_count += 1
            else:
                missing_count += 1
                missing_files.append(sar_filename)
        
        total_matched += matched_count
        total_missing += missing_count
        
        print(f"'{class_name}' 클래스 결과:")
        print(f"  - 매칭된 이미지 수: {matched_count}")
        print(f"  - 매칭되지 않은 이미지 수: {missing_count}")
        
        if missing_count > 0 and missing_count <= 10:
            print(f"  - 매칭되지 않은 파일: {missing_files}")
        elif missing_count > 10:
            print(f"  - 매칭되지 않은 파일 일부: {missing_files[:10]} ...")

    print("\n최종 결과:")
    print(f"총 매칭된 이미지 수: {total_matched}")
    print(f"총 매칭되지 않은 이미지 수: {total_missing}")
    
    if total_matched > 0:
        print(f"매칭 성공률: {(total_matched / (total_matched + total_missing)) * 100:.2f}%")
    
    print("EO 이미지 복사 완료!")

if __name__ == "__main__":
    copy_matching_eo_images()