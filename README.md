# Enhanced_SkinCancer_Classification
Hybrid CNN-Transformer skin cancer classifier achieving 87.93% accuracy with dual-backbone architecture (ConvNeXt V2 + EfficientNet V2), MSAF attention fusion, and knowledge distillation for 32× model compression. Improves melanoma detection from 21.9% to 85.8% precision on HAM10000 dataset. Production-ready with ONNX export and 120ms inference.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU)
- 16GB RAM minimum
- 50GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/skin-cancer-classifier.git
cd skin-cancer-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dataset Setup

### Step 1: Get Kaggle Credentials
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Download `kaggle.json`

### Step 2: Configure Kaggle
```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

### Step 3: Download Dataset
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d HAM10000/
```

## 🏃 Running the Project

### Phase 1: Data Preprocessing
```bash
python scripts/01_prepare_data.py
```
Output: `train_df.pkl`, `val_df.pkl`, `test_df.pkl`

### Phase 2: Train Teacher Model
```bash
python scripts/02_train_teacher.py --epochs 20 --batch-size 32 --lr 1e-4
```
Output: `checkpoints/teacher_best_224.pth`

### Phase 3: Train Student Model
```bash
python scripts/03_train_student.py --epochs 30 --batch-size 32 --lr 1e-4
```
Output: `checkpoints/student_best.pth`

### Phase 4: Evaluate Models
```bash
python scripts/04_evaluate.py
```
Output: `results/confusion_matrix.png`, `results/metrics.txt`

### Phase 5: Export to ONNX
```bash
python scripts/05_export_onnx.py
```
Output: `models/student_model.onnx`

### Phase 6: Run Inference
```bash
python scripts/06_inference.py --image path/to/lesion.jpg
```

## 📊 Expected Results

| Model | Accuracy | Melanoma Precision | Size | Inference Time |
|-------|----------|-------------------|------|----------------|
| Teacher | 87.93% | 85.8% | 480MB | 250ms |
| Student | 85.40% | 83.2% | 15MB | 120ms |

## 🛠️ Project Structure

```
skin-cancer-classifier/
├── data/
│   └── HAM10000/           # Dataset directory
├── models/
│   ├── teacher.py          # Teacher architecture
│   ├── student.py          # Student architecture
│   ├── msaf.py             # MSAF module
│   └── distillation.py     # Distillation loss
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_teacher.py
│   ├── 03_train_student.py
│   ├── 04_evaluate.py
│   ├── 05_export_onnx.py
│   └── 06_inference.py
├── utils/
│   ├── dataset.py          # Custom dataset class
│   ├── transforms.py       # Augmentations
│   └── metrics.py          # Evaluation metrics
├── checkpoints/            # Saved models
├── results/                # Evaluation outputs
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `config.py` to modify:
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 30
TEMPERATURE = 4.0
ALPHA = 0.7
```

## 📝 Requirements

```
torch==2.0.1
torchvision==0.15.2
timm==0.9.2
albumentations==1.3.1
opencv-python==4.8.0.76
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
onnx==1.14.0
onnxruntime==1.15.1
kaggle==1.5.16
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/02_train_teacher.py --batch-size 16
```

### Kaggle API Error
```bash
# Verify credentials
kaggle datasets list
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```


## 📧 Contact

- **Author:** Chetan Kumar Patruni
- **LinkedIn:** https://www.linkedin.com/in/chetan-kumar-patruni/

## 🙏 Acknowledgments

- Base paper: Jang & Park (2024) - DER Algorithm
- HAM10000 dataset: Tschandl et al. (2018)
- TIMM library maintainers
