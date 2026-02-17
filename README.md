# Heart Attack Prediction Using ECG and PPG Datasets

A comprehensive machine learning and deep learning pipeline for predicting heart attacks using ECG (Electrocardiogram) and PPG (Photoplethysmogram) physiological signals.

## 📊 Project Overview

This project implements a complete research-grade pipeline that progresses from basic preprocessing and exploratory data analysis through traditional ML models to advanced deep learning architectures, with extensive model comparison and interpretability analysis.

### Datasets
- **ECG**: 4,997 samples × 141 features (time-series signal data)
- **PPG**: 2,576 samples × 2,001 features (time-series signal data)

## 🚀 Features

### Data Preprocessing
- Signal filtering (Butterworth, Savitzky-Golay)
- Noise removal and baseline wander correction
- Normalization (z-score, min-max, robust scaling)
- Outlier detection and handling (IQR, isolation forest)
- Missing value imputation
- Class imbalance handling (SMOTE, ADASYN)

### Feature Extraction
- **Time-domain features**: Statistical, peak detection, morphological
- **Frequency-domain features**: FFT, PSD, spectral analysis
- **Wavelet features**: DWT, wavelet energy, wavelet entropy
- **HRV features**: Time-domain, frequency-domain, and non-linear HRV metrics

### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM
- CatBoost

### Deep Learning Models
- Multi-Layer Perceptron (MLP)
- 1D Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)

### Evaluation & Analysis
- Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Kappa)
- Cross-validation
- Model comparison and ranking
- Statistical significance testing

## 📁 Project Structure

```
ecg_ppg_paper/
├── data/
│   ├── processed/              # Preprocessed data
│   └── features/               # Extracted features
├── src/
│   ├── preprocessing/          # Data loading and preprocessing
│   │   ├── data_loader.py
│   │   ├── signal_processing.py
│   │   └── data_splitting.py
│   ├── features/               # Feature extraction
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   ├── wavelet_features.py
│   │   └── hrv_features.py
│   ├── models/                 # Model implementations
│   │   ├── traditional_ml.py
│   │   └── deep_learning.py
│   ├── evaluation/             # Evaluation utilities
│   │   ├── metrics.py
│   │   └── comparison.py
│   └── visualization/          # Visualization tools
├── results/
│   ├── figures/                # Generated plots
│   ├── models/                 # Saved models
│   └── reports/                # Analysis reports
├── main_pipeline.py            # Main pipeline script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for deep learning acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ecg_ppg_paper
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

Run the complete pipeline on ECG dataset:
```bash
python main_pipeline.py --dataset ecg
```

Run on PPG dataset:
```bash
python main_pipeline.py --dataset ppg
```

Run on both datasets:
```bash
python main_pipeline.py --dataset both
```

### Quick Test Mode

For rapid testing with reduced hyperparameter grids:
```bash
python main_pipeline.py --dataset ecg --quick-test
```

### Configuration

Edit `config.yaml` to customize:
- Data paths
- Preprocessing parameters
- Feature extraction settings
- Model hyperparameters
- Training configurations

### Individual Module Usage

**Load and preprocess data:**
```python
from src.preprocessing.data_loader import load_datasets
from src.preprocessing.signal_processing import preprocess_pipeline

datasets = load_datasets()
ecg_df, ppg_df = datasets['ecg'], datasets['ppg']
```

**Extract features:**
```python
from src.features.time_domain import extract_time_features_from_dataframe
from src.features.frequency_domain import extract_frequency_features_from_dataframe

time_features = extract_time_features_from_dataframe(ecg_df)
freq_features = extract_frequency_features_from_dataframe(ecg_df)
```

**Train models:**
```python
from src.models.traditional_ml import train_random_forest
from src.models.deep_learning import train_deep_learning_model, MLPClassifier

# Traditional ML
model, metrics = train_random_forest(X_train, y_train, X_val, y_val)

# Deep Learning
mlp = MLPClassifier(input_dim=100, num_classes=2)
trained_model, metrics = train_deep_learning_model(mlp, X_train, y_train, X_val, y_val)
```

**Evaluate and compare:**
```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.comparison import compare_models

results = evaluate_model(model, X_test, y_test)
comparison_df = compare_models(all_results)
```

## 📊 Results

After running the pipeline, results will be saved in:
- `results/models/`: Trained model files
- `results/reports/`: Model comparison tables and metrics
- `results/figures/`: Visualization plots

### Expected Performance
- All models should achieve >70% accuracy (baseline)
- Deep learning models typically outperform traditional ML
- Ensemble models (XGBoost, LightGBM, CatBoost) show strong performance
- Best models typically achieve 85-95% accuracy depending on dataset

## 🔧 Advanced Configuration

### Hyperparameter Tuning

Modify `config.yaml` to adjust hyperparameter search spaces:

```yaml
ml_models:
  random_forest:
    enabled: true
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, 30]
    min_samples_split: [2, 5]
```

### Feature Selection

Enable/disable feature types in `config.yaml`:

```yaml
features:
  time_domain: true
  frequency_domain: true
  wavelet: true
  hrv: true  # ECG only
```

### Class Imbalance Handling

Choose resampling strategy:

```yaml
imbalance:
  method: "smote"  # Options: smote, adasyn, class_weights, none
  sampling_strategy: "auto"
```

## 📈 Performance Benchmarks

Approximate execution times (with GPU):
- **Preprocessing**: < 2 minutes
- **Feature extraction**: < 5 minutes
- **Traditional ML training**: 1-5 minutes per model
- **Deep learning training**: 10-30 minutes per model
- **Full pipeline**: 1-4 hours (depending on configuration)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- ECG and PPG datasets from [source]
- Built with scikit-learn, PyTorch, XGBoost, and other open-source libraries

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{heart_attack_prediction,
  title={Heart Attack Prediction Using ECG and PPG Datasets},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/ecg_ppg_paper}
}
```

## 🔬 Research Applications

This pipeline can be used for:
- Cardiovascular disease prediction research
- Signal processing algorithm development
- Machine learning model benchmarking
- Feature engineering studies
- Deep learning architecture comparison

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size in `config.yaml`
- Use CPU instead: Set `device='cpu'` in deep learning training

**Missing dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Data loading errors:**
- Verify dataset paths in `config.yaml`
- Ensure CSV files are in the correct format

## 📊 Visualization

The pipeline generates various visualizations:
- Signal waveforms
- Feature distributions
- Correlation heatmaps
- ROC curves
- Confusion matrices
- Model comparison charts

## 🎯 Future Enhancements

- [ ] Add more deep learning architectures (Transformers, ResNet)
- [ ] Implement SHAP and LIME interpretability
- [ ] Add real-time prediction API
- [ ] Create interactive dashboard
- [ ] Add more visualization options
- [ ] Implement ensemble learning strategies
- [ ] Add support for more datasets
