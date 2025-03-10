# PBVS 2025 Challenge: SAR-EO Classification with Knowledge Distillation

## Overview
This repository implements a SAR-EO image classification model for the PBVS 2025 Challenge. We enhance SAR image classification performance by combining a ResNet101-based model with Knowledge Distillation, Feature Matching, and Contrastive Learning.

## Key Features
- **Knowledge Distillation**: Transfers knowledge from EO to SAR domain
- **Confidence Prediction**: Estimates model uncertainty to provide classification confidence
- **Feature Matching**: Reduces domain gap through SAR-EO feature alignment
- **Contrastive Learning**: Learns robust feature representations
- **EO Pre-training**: Leverages weights pre-trained on EO images

## Environment Setup
```bash
# Install required packages
pip install torch torchvision tqdm scikit-learn matplotlib tensorboard pandas
```

## Dataset Structure
The dataset should be organized as follows:
```
/path/to/dataset/
├── train/
│   ├── sar/
│   │   ├── class_0/
│   │   ├── class_1/
│   │   └── ...
│   └── eo/
│       ├── class_0/
│       ├── class_1/
│       └── ...
└── val/
    ├── sar/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ...
    └── eo/
        ├── class_0/
        ├── class_1/
        └── ...
```

## Pre-trained Models
Pre-trained models are available at the following Google Drive link:
[Download Pre-trained Models](https://drive.google.com/drive/folders/your-folder-id)

Save the downloaded model files to the `pretrained/` directory.

## Training
Training is performed using the `train_model.py` script with the shell script `mmd_con.sh`:

```bash
# Run training with all features
./shell/mmd_con.sh
```

The `mmd_con.sh` script contains the following configuration:

```bash
python ../train_model.py \
--data_root /path/to/dataset \
--output_dir ../runs/all_resnet101_eo_pretrained_mmdcon_confi_4_100ep_1e-3 \
--batch_size 128 \
--pretrained_path ./pretrained/eo_pretrained.pth \
--epochs 100 \
--lr 1e-3 \
--optimizer adamw \
--use_feature_matching \
--feature_matching_weight 0.1 \
--contrastive_method supcon \
--contrastive_weight 0.1 \
--temperature 0.3 \
--gpus "0,1,2" \
--use_class_weights \
--use_confidence \
--confidence_lambda 1.0
```

### Key Training Parameters
- `--data_root`: Dataset path
- `--output_dir`: Results save path
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--optimizer`: Optimization algorithm (adam, adamw, sgd)
- `--pretrained_path`: EO pre-trained model path
- `--use_feature_matching`: Use Feature Matching Loss
- `--feature_matching_weight`: Feature Matching Loss weight
- `--contrastive_method`: Contrastive Learning method (supcon, simclr)
- `--contrastive_weight`: Contrastive Loss weight
- `--temperature`: Contrastive Learning temperature parameter
- `--use_confidence`: Use Confidence prediction
- `--confidence_lambda`: Confidence Loss weight
- `--gpus`: List of GPUs to use

## Inference
Inference is performed using the `test.py` script:

```bash
python test.py \
  --model_path ./runs/all_resnet101_eo_pretrained_mmdcon_confi_4_100ep_1e-3/best_model.pth \
  --test_dir /path/to/test/data \
  --output_file ./results/predictions.csv \
  --batch_size 64 \
  --gpu 0
```

### Inference Parameters
- `--model_path`: Path to the trained model
- `--test_dir`: Directory containing test SAR images
- `--output_file`: Path to save prediction results
- `--batch_size`: Batch size for inference
- `--gpu`: GPU ID to use

## Model Architecture
This implementation is based on ResNet101 with the following structure:
- SAR image encoder: ResNet101
- EO image encoder: ResNet101 (pre-trained)
- Confidence prediction head: 2-layer MLP
- Projection head: 2-layer MLP (for Contrastive Learning)

## Citation
If you use this code, please cite:
```
@article{min2025pbvs,
  title={SAR-EO Classification with Knowledge Distillation for PBVS 2025 Challenge},
  author={Min, Jeongho},
  year={2025}
}
```

## License
MIT License
