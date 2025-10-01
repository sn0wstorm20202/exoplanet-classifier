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
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV,
    StratifiedKFold, 
    cross_val_score,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    make_scorer,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.feature_selection import SelectFromModel, RFECV
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
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Find optimal hyperparameters using Grid Search"""
        
        logger.info("\n=== HYPERPARAMETER OPTIMIZATION ===")
        logger.info("This may take 5-15 minutes...")
        
        # Define parameter grid (focused and bounded for runtime)
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [15, 18, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Use stratified k-fold for class balance (deeper search)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Optimize for F1-score (better for imbalanced classes)
        scorer = make_scorer(f1_score, average='weighted')
        
        # Initialize base model
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Randomized search to bound runtime
        logger.info("Running randomized search (n_iter=60, cv=5)...")
        rand_search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_grid,
            n_iter=60,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        rand_search.fit(X_train, y_train)
        
        logger.info(f"\n✅ Best parameters found:")
        for param, value in rand_search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"\nBest cross-validation F1-score: {rand_search.best_score_:.4f}")
        
        return rand_search.best_estimator_, rand_search.best_params_
    
    def create_ensemble_model(self, X_train, y_train, best_rf_params=None):
        """Create ensemble of multiple classifiers for better accuracy"""
        
        logger.info("\n=== CREATING ENSEMBLE MODEL ===")
        
        # Model 1: Optimized Random Forest
        if best_rf_params:
            rf_params = {**best_rf_params, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
        else:
            rf_params = {
                'n_estimators': 300,
                'max_depth': 18,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        
        rf = RandomForestClassifier(**rf_params)
        
        # Model 2: Gradient Boosting (good for sequential patterns)
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Ensemble: Soft voting (use probability averages)
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            weights=[2, 1]  # Give RF more weight
        )
        
        logger.info("Training ensemble model...")
        start_time = datetime.now()
        ensemble.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Ensemble trained in {training_time:.2f} seconds")
        
        return ensemble
    
    def handle_class_imbalance(self, X_train, y_train):
        """Handle class imbalance using class weights (simpler approach)"""
        
        logger.info("\n=== ANALYZING CLASS IMBALANCE ===")
        
        # Count classes
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique, counts):
            label_name = self.label_encoder.classes_[label] if hasattr(self, 'label_encoder') else label
            percentage = (count / len(y_train)) * 100
            logger.info(f"  Class {label} ({label_name}): {count} ({percentage:.1f}%)")
        
        # Check if imbalance is significant
        imbalance_ratio = counts.max() / counts.min()
        logger.info(f"\nImbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            logger.info(f"Significant imbalance detected (ratio > 2.0)")
            logger.info(f"Will use class_weight='balanced' in models to handle this")
        else:
            logger.info(f"Classes relatively balanced (ratio ≤ 2.0)")
        
        # For now, return original data and rely on class_weight='balanced' in models
        logger.info("Using class_weight='balanced' approach for imbalance handling")
        
        return X_train, y_train
    
    def select_best_features(self, X_train, y_train, feature_names):
        """Select most important features using multiple methods"""
        
        logger.info("\n=== FEATURE SELECTION ===")
        
        # Method 1: RandomForest feature importance
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_selector.fit(X_train, y_train)
        
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 15 most important features:")
        for idx, row in importances.head(15).iterrows():
            logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        # Store feature importance for metadata
        self.feature_importance_ranking = importances.to_dict('records')
        
        # Select features with importance > threshold (keep top 80% of importance)
        cumulative_importance = importances['importance'].cumsum() / importances['importance'].sum()
        selected_indices = cumulative_importance <= 0.95  # Keep features contributing to 95% of importance
        
        selected_features = importances.loc[selected_indices, 'feature'].tolist()
        
        logger.info(f"\n✅ Selected {len(selected_features)} features (top 95% importance)")
        logger.info(f"   Reduced from {len(feature_names)} to {len(selected_features)} features")
        
        return selected_features, importances
    
    def train_model(self, X_train, y_train, use_optimization=True, use_ensemble=True):
        """Train optimized classifier with advanced techniques"""
        
        logger.info("\n=== ADVANCED MODEL TRAINING ===")
        
        if use_optimization:
            # Find optimal hyperparameters
            try:
                optimized_model, best_params = self.optimize_hyperparameters(X_train, y_train)
                logger.info("Using optimized hyperparameters")
            except BaseException as e:
                logger.warning(f"Hyperparameter optimization skipped/failed: {e}")
                logger.info("Falling back to default parameters")
                optimized_model = None
                best_params = None
        else:
            optimized_model = None
            best_params = None
        
        if use_ensemble and optimized_model:
            # Create ensemble with optimized parameters
            self.classifier = self.create_ensemble_model(X_train, y_train, best_params)
        elif use_ensemble:
            # Create ensemble with default parameters
            self.classifier = self.create_ensemble_model(X_train, y_train)
        elif optimized_model:
            # Use optimized single model
            self.classifier = optimized_model
        else:
            # Fallback to basic RandomForest
            logger.info("🌲 Training basic RandomForest classifier...")
            rf_params = {
                'n_estimators': 300,
                'max_depth': 18,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            
            self.classifier = RandomForestClassifier(**rf_params)
            
            start_time = datetime.now()
            self.classifier.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
        
        return self.classifier
    
    def evaluate_with_cross_validation(self, X, y, cv=5):
        """Perform comprehensive k-fold cross-validation"""
        
        logger.info(f"\n=== {cv}-FOLD CROSS-VALIDATION ===")
        
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted'
        }
        
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_validate(
            self.classifier, X, y,
            cv=cv_obj,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        logger.info("Cross-validation results:")
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            train_scores = scores[f'train_{metric}']
            test_scores = scores[f'test_{metric}']
            
            logger.info(f"\n{metric}:")
            logger.info(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
            logger.info(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
        
        return scores
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation with advanced metrics"""
        
        logger.info("\n=== COMPREHENSIVE MODEL EVALUATION ===")
        
        # Predictions
        y_train_pred = self.classifier.predict(X_train)
        y_test_pred = self.classifier.predict(X_test)
        
        # Get prediction probabilities for additional metrics
        y_test_pred_proba = self.classifier.predict_proba(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate additional metrics
        precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        
        logger.info(f"🎯 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        logger.info(f"🎯 Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"🎯 Weighted Precision: {precision_weighted:.4f}")
        logger.info(f"🎯 Weighted Recall: {recall_weighted:.4f}")
        logger.info(f"🎯 Weighted F1-Score: {f1_weighted:.4f}")
        
        # Cross-validation on full training set
        cv_scores = self.evaluate_with_cross_validation(X_train, y_train)
        cv_accuracy_mean = cv_scores['test_accuracy'].mean()
        cv_accuracy_std = cv_scores['test_accuracy'].std()
        cv_f1_mean = cv_scores['test_f1_weighted'].mean()
        cv_f1_std = cv_scores['test_f1_weighted'].std()
        
        logger.info(f"\n🔄 Cross-validation Accuracy: {cv_accuracy_mean:.4f} ± {cv_accuracy_std:.4f}")
        logger.info(f"🔄 Cross-validation F1-Score: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        classification_rep = classification_report(
            y_test, y_test_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        logger.info(f"\n📈 DETAILED CLASSIFICATION REPORT:")
        for class_name in class_names:
            metrics = classification_rep[class_name]
            support = int(metrics['support'])
            logger.info(f"\n  {class_name} (n={support}):")
            logger.info(f"    Precision: {metrics['precision']:.3f}")
            logger.info(f"    Recall: {metrics['recall']:.3f}")
            logger.info(f"    F1-score: {metrics['f1-score']:.3f}")
        
        # Confusion matrix with percentages
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        logger.info(f"\n🔲 CONFUSION MATRIX (counts and percentages):")
        logger.info(f"     Predicted: {' '.join([f'{name:>12}' for name in class_names])}")
        for i, true_name in enumerate(class_names):
            row_counts = ' '.join([f'{conf_matrix[i][j]:>12}' for j in range(len(class_names))])
            row_pcts = ' '.join([f'{conf_matrix_pct[i][j]:>11.1f}%' for j in range(len(class_names))])
            logger.info(f"Actual {true_name:>8}: {row_counts}")
            logger.info(f"      (%)     : {row_pcts}")
        
        # Feature importance with enhanced analysis
        if hasattr(self.classifier, 'feature_importances_'):
            feature_importance = self.classifier.feature_importances_
        elif hasattr(self.classifier, 'estimators_'):  # For ensemble models
            # Average importance across ensemble members
            if hasattr(self.classifier.estimators_[0], 'feature_importances_'):
                importances = [est.feature_importances_ for est in self.classifier.estimators_]
                feature_importance = np.mean(importances, axis=0)
            else:
                feature_importance = np.zeros(len(self.feature_names))
        else:
            feature_importance = np.zeros(len(self.feature_names))
        
        # Create detailed feature importance analysis
        feature_importance_list = []
        for i, (feature, importance) in enumerate(zip(self.feature_names, feature_importance)):
            feature_importance_list.append({
                'feature': feature,
                'importance': float(importance),
                'rank': i + 1
            })
        
        # Sort by importance
        feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        logger.info(f"\n🌟 TOP 15 FEATURE IMPORTANCE RANKINGS:")
        for i, feat_info in enumerate(feature_importance_list[:15]):
            logger.info(f"  {i+1:2d}. {feat_info['feature']:25s}: {feat_info['importance']:.4f}")
        
        # Identify high-value feature categories
        quality_features = [f for f in feature_importance_list[:10] if any(x in f['feature'] for x in ['snr', 'fp_flag', 'score', 'signal'])]
        physical_features = [f for f in feature_importance_list[:10] if any(x in f['feature'] for x in ['density', 'ratio', 'hz_', 'mass', 'eqt'])]
        
        if quality_features:
            logger.info(f"\n🔥 HIGH-IMPACT QUALITY FEATURES:")
            for feat in quality_features[:5]:
                logger.info(f"  {feat['feature']:25s}: {feat['importance']:.4f}")
        
        if physical_features:
            logger.info(f"\n⭐ HIGH-IMPACT PHYSICAL FEATURES:")
            for feat in physical_features[:5]:
                logger.info(f"  {feat['feature']:25s}: {feat['importance']:.4f}")
        
        # Store comprehensive evaluation results
        self.evaluation_results = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'cv_accuracy_mean': float(cv_accuracy_mean),
            'cv_accuracy_std': float(cv_accuracy_std),
            'cv_f1_mean': float(cv_f1_mean),
            'cv_f1_std': float(cv_f1_std),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_percentages': conf_matrix_pct.tolist(),
            'feature_importance': {feat['feature']: feat['importance'] for feat in feature_importance_list},
            'feature_importance_ranking': feature_importance_list,
            'quality_features': [feat['feature'] for feat in quality_features],
            'physical_features': [feat['feature'] for feat in physical_features]
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
        cv_mean = self.evaluation_results.get('cv_accuracy_mean', self.evaluation_results.get('cv_mean', test_accuracy))
        
        print(f"\n{'='*60}")
        print(f"🎯 MODEL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        cv_std = self.evaluation_results.get('cv_accuracy_std', self.evaluation_results.get('cv_std', 0.0))
        print(f"✅ Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
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
    """Enhanced main training function with advanced ML techniques"""
    print("🚀 ENHANCED Exoplanet Classification Model Training")
    print("="*70)
    print("Advanced features: Hyperparameter optimization, Ensemble methods,")
    print("Feature selection, Class balancing, Comprehensive evaluation")
    print("="*70)
    
    try:
        # Initialize trainer
        trainer = ExoplanetModelTrainer()
        
        # Load and validate data
        df, feature_names, label_column = trainer.load_processed_data()
        logger.info(f"📊 Loaded {len(feature_names)} features: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
        
        # Prepare features and labels
        X_train, X_test, y_train, y_test = trainer.prepare_features_and_labels()
        logger.info(f"🔄 Initial training set: {X_train.shape}")
        
        # Skip feature selection for stability - use all 27 features
        logger.info(f"🎯 Using all {len(feature_names)} features for maximum impact")
        
        # Scale features
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(X_train_scaled, y_train)
        logger.info(f"⚖️ Balanced training set: {X_train_balanced.shape}")
        
        # Train advanced model with optimization and ensemble
        logger.info("\n" + "="*50)
        logger.info("🎯 STARTING ADVANCED MODEL TRAINING")
        logger.info("="*50)
        
        # Configuration: enable optimization (bounded) + ensemble for best accuracy
        USE_HYPERPARAMETER_OPTIMIZATION = True
        USE_ENSEMBLE = True
        
        classifier = trainer.train_model(
            X_train_balanced, 
            y_train_balanced, 
            use_optimization=USE_HYPERPARAMETER_OPTIMIZATION,
            use_ensemble=USE_ENSEMBLE
        )
        
        logger.info("\n" + "="*50)
        logger.info("📊 COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*50)
        
        # Evaluate model on original (unbalanced) test set for realistic performance
        results = trainer.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save enhanced model
        trainer.save_model()
        
        # Print success summary
        print(f"\n{'='*70}")
        print(f"🎉 ENHANCED TRAINING COMPLETE - PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        test_acc = results['test_accuracy']
        cv_acc = results['cv_accuracy_mean']
        f1_score = results['f1_weighted']
        
        print(f"🎯 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"🔄 CV Accuracy: {cv_acc:.4f} ± {results['cv_accuracy_std']:.4f}")
        print(f"📊 F1-Score: {f1_score:.4f}")
        print(f"🔬 Features used: {len(trainer.feature_names)}")
        print(f"🧪 Training samples: {X_train_balanced.shape[0]}")
        
        # Performance assessment
        if test_acc >= 0.95:
            print(f"🌟 OUTSTANDING performance! (95%+)")
        elif test_acc >= 0.92:
            print(f"⭐ EXCELLENT performance! (92%+)")
        elif test_acc >= 0.90:
            print(f"✨ Very good performance! (90%+)")
        elif test_acc >= 0.85:
            print(f"👍 Good performance (85%+)")
        else:
            print(f"⚠️  Performance could be improved (<85%)")
        
        # Highlight key improvements
        quality_features = results.get('quality_features', [])
        if quality_features:
            print(f"\n🔥 Quality indicators working: {len(quality_features)} features")
            print(f"   Examples: {', '.join(quality_features[:3])}")
        
        print(f"\n💾 Model saved to: {trainer.models_dir}")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {str(e)}")
        logger.error("Attempting fallback to basic training...")
        
        try:
            # Fallback to basic training
            trainer = ExoplanetModelTrainer()
            df, feature_names, label_column = trainer.load_processed_data()
            X_train, X_test, y_train, y_test = trainer.prepare_features_and_labels()
            X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
            
            # Basic training without optimization
            classifier = trainer.train_model(
                X_train_scaled, y_train, 
                use_optimization=False, use_ensemble=False
            )
            results = trainer.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
            trainer.save_model()
            
            logger.info("✅ Fallback training completed successfully")
            
        except Exception as fallback_error:
            logger.error(f"❌ Fallback training also failed: {str(fallback_error)}")
            raise

if __name__ == "__main__":
    main()