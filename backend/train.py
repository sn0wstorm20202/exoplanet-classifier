#!/usr/bin/env python3
"""
Exoplanet Classification Model Training Script

This script trains a RandomForest classifier for exoplanet classification using
processed NASA datasets. It includes:
1. Data loading and validation
2. Feature scaling with StandardScaler
3. RandomForest training with optimized parameters
4. Model evaluation and metrics
5. Model persistence (classifier, scaler, label encoder, metadata)
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExoplanetModelTrainer:
    """Exoplanet classification model trainer."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.metadata = {}
        
        # Set up paths
        self.script_dir = Path(__file__).parent.parent
        self.data_dir = self.script_dir / 'data'
        self.models_dir = self.script_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
    def load_processed_data(self):
        """Load and validate processed data."""
        logger.info("🔄 Loading processed data...")
        
        # Load processed data
        data_path = self.data_dir / 'processed_combined.csv'
        feature_info_path = self.data_dir / 'feature_info.json'
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {data_path}\n"
                f"Please run notebooks/load_combined_data.py first"
            )
        
        if not feature_info_path.exists():
            raise FileNotFoundError(
                f"Feature info not found: {feature_info_path}\n"
                f"Please run notebooks/load_combined_data.py first"
            )
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Load feature information
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        
        self.feature_names = feature_info['features']
        self.label_column = feature_info['label_column']
        
        logger.info(f"✅ Loaded data shape: {self.df.shape}")
        logger.info(f"📋 Features: {self.feature_names}")
        logger.info(f"🏷️ Label column: {self.label_column}")
        
        # Validate data
        self._validate_data()
        
        return self.df, self.feature_names, self.label_column
    
    def _validate_data(self):
        """Validate loaded data for training."""
        logger.info("🔍 Validating data...")
        
        # Check for required columns
        missing_features = [f for f in self.feature_names if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        if self.label_column not in self.df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in data")
        
        # Check data types and missing values
        X = self.df[self.feature_names]
        y = self.df[self.label_column]
        
        logger.info(f"📊 Feature statistics:")
        for feature in self.feature_names:
            missing_count = X[feature].isnull().sum()
            data_type = X[feature].dtype
            logger.info(f"  {feature}: {data_type}, missing: {missing_count}")
        
        # Check label distribution
        label_counts = y.value_counts()
        logger.info(f"🏷️ Label distribution:")
        for label, count in label_counts.items():
            percentage = count / len(y) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Check for minimum class size
        min_class_size = label_counts.min()
        if min_class_size < 10:
            logger.warning(f"⚠️ Small class detected with only {min_class_size} samples")
        
        total_samples = len(self.df)
        logger.info(f"✅ Data validation complete: {total_samples} samples")
    
    def prepare_features_and_labels(self):
        """Prepare features and labels for training."""
        logger.info("🔧 Preparing features and labels...")
        
        # Extract features and labels
        X = self.df[self.feature_names].copy()
        y = self.df[self.label_column].copy()
        
        # Handle any remaining missing values
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"📝 Filling {missing_counts.sum()} missing values with median")
            X = X.fillna(X.median())
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"🏷️ Label encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            logger.info(f"  {label} → {i}")
        
        # Split data
        test_size = 0.2
        random_state = 42
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        logger.info(f"📊 Data split:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        logger.info("⚖️ Scaling features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Log scaling statistics
        logger.info(f"📏 Feature scaling statistics:")
        for i, feature in enumerate(self.feature_names):
            mean = self.scaler.mean_[i]
            scale = self.scaler.scale_[i]
            logger.info(f"  {feature}: mean={mean:.3f}, scale={scale:.3f}")
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train):
        """Train RandomForest classifier."""
        logger.info("🌲 Training RandomForest classifier...")
        
        # RandomForest parameters
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        logger.info(f"🔧 RandomForest parameters: {rf_params}")
        
        # Initialize and train classifier
        self.classifier = RandomForestClassifier(**rf_params)
        
        # Train model
        start_time = datetime.now()
        self.classifier.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
        
        return self.classifier
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance."""
        logger.info("📊 Evaluating model performance...")
        
        # Predictions
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"🎯 Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"🎯 Testing Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"🔄 Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        classification_rep = classification_report(
            y_test, y_test_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        logger.info(f"📈 Classification Report:")
        for class_name in class_names:
            metrics = classification_rep[class_name]
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    F1-score: {metrics['f1-score']:.3f}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        logger.info(f"🔲 Confusion Matrix:")
        logger.info(f"     Predicted: {' '.join([f'{name:>12}' for name in class_names])}")
        for i, true_name in enumerate(class_names):
            row = ' '.join([f'{conf_matrix[i][j]:>12}' for j in range(len(class_names))])
            logger.info(f"Actual {true_name:>8}: {row}")
        
        # Feature importance
        feature_importance = self.classifier.feature_importances_
        feature_importance_dict = dict(zip(self.feature_names, feature_importance))
        
        logger.info(f"🌟 Feature Importance:")
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            logger.info(f"  {feature}: {importance:.4f}")
        
        # Store evaluation results
        self.evaluation_results = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean': float(cv_mean),
            'cv_std': float(cv_std),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance_dict
        }
        
        return self.evaluation_results
    
    def save_model(self):
        """Save trained model and components."""
        logger.info("💾 Saving model components...")
        
        # Create metadata
        self.metadata = {
            'model_type': 'RandomForestClassifier',
            'training_timestamp': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'label_classes': self.label_encoder.classes_.tolist(),
            'data_shape': self.df.shape,
            'model_parameters': self.classifier.get_params(),
            'evaluation_results': self.evaluation_results,
            'version': '1.0.0'
        }
        
        # Save model components
        model_files = {
            'classifier.pkl': self.classifier,
            'scaler.pkl': self.scaler,
            'label_encoder.pkl': self.label_encoder
        }
        
        for filename, component in model_files.items():
            filepath = self.models_dir / filename
            with open(filepath, 'wb') as f:
                pickle.dump(component, f)
            logger.info(f"✅ Saved {filename}")
        
        # Save metadata
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"✅ Saved model_metadata.json")
        
        logger.info(f"🎉 Model saved successfully to {self.models_dir}")
        
        # Summary
        test_accuracy = self.evaluation_results['test_accuracy']
        cv_mean = self.evaluation_results['cv_mean']
        
        print(f"\n{'='*60}")
        print(f"🎯 MODEL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"✅ Cross-validation: {cv_mean:.4f} ± {self.evaluation_results['cv_std']:.4f}")
        print(f"✅ Model saved to: {self.models_dir}")
        print(f"✅ Features used: {len(self.feature_names)}")
        print(f"✅ Training samples: {self.df.shape[0]}")
        
        # Performance assessment
        if test_accuracy >= 0.90:
            print(f"🌟 Excellent performance!")
        elif test_accuracy >= 0.85:
            print(f"✨ Good performance!")
        elif test_accuracy >= 0.75:
            print(f"👍 Acceptable performance")
        else:
            print(f"⚠️  Performance may need improvement")
        
        print(f"{'='*60}")

def main():
    """Main training function."""
    print("🚀 Exoplanet Classification Model Training")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = ExoplanetModelTrainer()
        
        # Load and validate data
        df, feature_names, label_column = trainer.load_processed_data()
        
        # Prepare features and labels
        X_train, X_test, y_train, y_test = trainer.prepare_features_and_labels()
        
        # Scale features
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Train model
        classifier = trainer.train_model(X_train_scaled, y_train)
        
        # Evaluate model
        results = trainer.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save model
        trainer.save_model()
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()