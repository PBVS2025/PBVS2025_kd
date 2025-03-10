import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset, Dataset

def balance_dataset(dataset: Dataset, num_samples_per_class=50, seed=42):
    """
    주어진 dataset에서 클래스별로 num_samples_per_class만큼 샘플링하여 새로운 균형 데이터셋을 생성합니다.

    Args:
        dataset (Dataset): PyTorch 데이터셋 (예: ImageFolder, Custom Dataset)
        num_samples_per_class (int): 각 클래스당 포함할 샘플 수
        seed (int): 랜덤 시드 값 (기본값=42)

    Returns:
        Subset: 균형을 맞춘 데이터셋의 서브셋
    """
    random.seed(seed)  # Python random seed 고정
    np.random.seed(seed)  # NumPy random seed 고정

    class_to_indices = defaultdict(list)

    # 각 클래스별 인덱스 수집
    for idx, (sar, eo, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    balanced_indices = []

    # 각 클래스별 샘플링
    for label, indices in class_to_indices.items():
        if len(indices) >= num_samples_per_class:
            balanced_indices.extend(random.sample(indices, num_samples_per_class))  # 랜덤 샘플링
        else:
            # 부족한 경우 중복 샘플링 (seed가 고정되므로 항상 같은 샘플이 반복됨)
            balanced_indices.extend(
                indices * (num_samples_per_class // len(indices)) +
                random.sample(indices, num_samples_per_class % len(indices))
            )

    return Subset(dataset, balanced_indices)