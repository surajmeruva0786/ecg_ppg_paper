"""
Quick start script to test the pipeline with a small subset of data.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """
    Run a quick test of the pipeline with minimal configuration.
    """
    logger.info("Starting Quick Test of Heart Attack Prediction Pipeline")
    logger.info("="*80)
    
    try:
        # Import main pipeline
        from main_pipeline import run_pipeline
        
        # Run with quick test mode
        run_pipeline(
            config_path='config.yaml',
            dataset_type='ecg',  # Start with ECG only
            quick_test=True      # Use reduced parameters for speed
        )
        
        logger.info("\n" + "="*80)
        logger.info("Quick Test Completed Successfully!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Check results/reports/ecg/model_comparison.csv for model performance")
        logger.info("2. Review pipeline.log for detailed execution logs")
        logger.info("3. Run full pipeline: python main_pipeline.py --dataset ecg")
        
    except Exception as e:
        logger.error(f"Error during quick test: {str(e)}")
        logger.error("Please check that:")
        logger.error("1. All dependencies are installed: pip install -r requirements.txt")
        logger.error("2. Dataset files (ecg.csv, PPG_Dataset.csv) are in the correct location")
        logger.error("3. config.yaml is properly configured")
        raise

if __name__ == "__main__":
    quick_test()
