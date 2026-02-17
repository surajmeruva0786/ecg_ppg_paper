"""
Main pipeline script for heart attack prediction.

This script orchestrates the entire pipeline from data loading to model evaluation.
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from preprocessing.data_loader import load_datasets, load_config
from preprocessing.signal_processing import preprocess_pipeline
from preprocessing.data_splitting import split_data, apply_smote
from features.time_domain import extract_time_features_from_dataframe
from features.frequency_domain import extract_frequency_features_from_dataframe
from features.wavelet_features import extract_wavelet_features_from_dataframe
from features.hrv_features import extract_hrv_features_from_dataframe
from models.traditional_ml import train_all_ml_models
from models.deep_learning import train_all_dl_models
from evaluation.metrics import evaluate_model
from evaluation.comparison import compare_models, create_comparison_table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_all_features(df, config, dataset_name='Dataset'):
    """
    Extract all features from a dataset.
    """
    logger.info(f"Extracting features from {dataset_name}...")
    
    all_features = []
    
    # Time-domain features
    if config['features']['time_domain']:
        logger.info("Extracting time-domain features...")
        time_features = extract_time_features_from_dataframe(df)
        all_features.append(time_features)
    
    # Frequency-domain features
    if config['features']['frequency_domain']:
        logger.info("Extracting frequency-domain features...")
        freq_features = extract_frequency_features_from_dataframe(df)
        all_features.append(freq_features)
    
    # Wavelet features
    if config['features']['wavelet']:
        logger.info("Extracting wavelet features...")
        wavelet_features = extract_wavelet_features_from_dataframe(df)
        all_features.append(wavelet_features)
    
    # HRV features (for ECG only)
    if config['features']['hrv'] and 'ecg' in dataset_name.lower():
        logger.info("Extracting HRV features...")
        hrv_features = extract_hrv_features_from_dataframe(df)
        all_features.append(hrv_features)
    
    # Combine all features
    if len(all_features) > 0:
        # Merge all feature dataframes
        combined_features = all_features[0]
        for feat_df in all_features[1:]:
            # Remove duplicate label columns
            feat_df_no_label = feat_df.drop(columns=['label'], errors='ignore')
            combined_features = pd.concat([combined_features, feat_df_no_label], axis=1)
        
        logger.info(f"Total features extracted: {combined_features.shape[1]}")
        return combined_features
    else:
        logger.warning("No features extracted!")
        return df


def run_pipeline(config_path='config.yaml', dataset_type='ecg', quick_test=False):
    """
    Run the complete pipeline.
    
    Args:
        config_path: Path to configuration file
        dataset_type: Which dataset to use ('ecg', 'ppg', or 'both')
        quick_test: If True, use smaller parameter grids for quick testing
    """
    logger.info("="*80)
    logger.info("Starting Heart Attack Prediction Pipeline")
    logger.info("="*80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Step 1: Load data
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Data")
    logger.info("="*80)
    
    datasets = load_datasets(config, dataset_type=dataset_type)
    
    # Process each dataset
    for dataset_name, df in datasets.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {dataset_name.upper()} Dataset")
        logger.info(f"{'='*80}")
        
        # Step 2: Preprocessing
        logger.info("\nStep 2: Preprocessing")
        df_processed = preprocess_pipeline(df, config, dataset_name)
        
        # Step 3: Feature Extraction
        logger.info("\nStep 3: Feature Extraction")
        df_features = extract_all_features(df_processed, config, dataset_name)
        
        # Separate features and labels
        X = df_features.drop(columns=['label'], errors='ignore')
        y = df_features['label'] if 'label' in df_features.columns else df_processed.iloc[:, -1]
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label distribution:\n{y.value_counts()}")
        
        # Step 4: Data Splitting
        logger.info("\nStep 4: Data Splitting")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y,
            train_size=config['split']['train_size'],
            val_size=config['split']['val_size'],
            test_size=config['split']['test_size'],
            random_state=config['random_seed'],
            stratify=config['split']['stratify']
        )
        
        # Step 5: Handle Class Imbalance
        if config['imbalance']['method'] == 'smote':
            logger.info("\nStep 5: Applying SMOTE")
            X_train, y_train = apply_smote(X_train, y_train, random_state=config['random_seed'])
        
        # Step 6: Train Traditional ML Models
        logger.info("\n" + "="*80)
        logger.info("Step 6: Training Traditional ML Models")
        logger.info("="*80)
        
        ml_results = train_all_ml_models(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            save_dir=f'results/models/{dataset_name}/ml',
            cv=3 if quick_test else config['evaluation']['cv_folds'],
            random_state=config['random_seed']
        )
        
        # Step 7: Train Deep Learning Models
        logger.info("\n" + "="*80)
        logger.info("Step 7: Training Deep Learning Models")
        logger.info("="*80)
        
        dl_results = train_all_dl_models(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            save_dir=f'results/models/{dataset_name}/dl',
            epochs=10 if quick_test else config['dl_models']['mlp']['epochs'],
            batch_size=config['dl_models']['mlp']['batch_size'],
            learning_rate=config['dl_models']['mlp']['learning_rate']
        )
        
        # Step 8: Evaluate on Test Set
        logger.info("\n" + "="*80)
        logger.info("Step 8: Final Evaluation on Test Set")
        logger.info("="*80)
        
        all_results = {**ml_results, **dl_results}
        test_results = {}
        
        for model_name, result in all_results.items():
            if 'model' in result:
                logger.info(f"\nEvaluating {model_name} on test set...")
                try:
                    # For PyTorch models
                    if hasattr(result['model'], 'eval'):
                        import torch
                        model = result['model']
                        model.eval()
                        with torch.no_grad():
                            X_test_tensor = torch.FloatTensor(X_test.values)
                            outputs = model(X_test_tensor)
                            _, y_pred = torch.max(outputs, 1)
                            y_pred = y_pred.numpy()
                            y_proba = torch.softmax(outputs, dim=1)[:, 1].numpy()
                    else:
                        # For sklearn models
                        y_pred = result['model'].predict(X_test.values)
                        y_proba = result['model'].predict_proba(X_test.values)[:, 1] if hasattr(result['model'], 'predict_proba') else None
                    
                    from evaluation.metrics import calculate_metrics
                    test_metrics = calculate_metrics(y_test.values, y_pred, y_proba)
                    test_results[model_name] = {'metrics': test_metrics}
                    
                    logger.info(f"{model_name} Test F1: {test_metrics['f1']:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        # Step 9: Model Comparison
        logger.info("\n" + "="*80)
        logger.info("Step 9: Model Comparison")
        logger.info("="*80)
        
        comparison_df = compare_models(test_results)
        print("\n" + str(comparison_df))
        
        # Save comparison table
        results_dir = Path(f'results/reports/{dataset_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_file = results_dir / 'model_comparison.csv'
        create_comparison_table(comparison_df, save_path=comparison_file)
        
        logger.info(f"\nComparison table saved to {comparison_file}")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline Completed Successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Heart Attack Prediction Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='ecg', choices=['ecg', 'ppg', 'both'], help='Dataset to use')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    run_pipeline(
        config_path=args.config,
        dataset_type=args.dataset,
        quick_test=args.quick_test
    )
