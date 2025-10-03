#!/usr/bin/env python3
"""
Exoplanet Classification Model Training Script

This script trains an XGBoost classifier for exoplanet classification using
processed NASA datasets. It includes:
1. Data loading and validation
2. Feature scaling with StandardScaler
3. XGBoost training with optimized parameters
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
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings

def diagnose_data(X_train, X_test, y_train, y_test):
    """Check for data quality issues"""
    
    print("\n" + "="*60)
    print("DATA DIAGNOSTIC")
    print("="*60)
    
    # Dataset sizes
    print(f"\nDataset sizes:")
    print(f"  Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    
    # Class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"\nClass distribution (Training):")
    for label, count in zip(unique_train, counts_train):
        pct = (count / len(y_train)) * 100
        print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nClass distribution (Test):")
    for label, count in zip(unique_test, counts_test):
        pct = (count / len(y_test)) * 100
        print(f"  Class {label}: {count:4d} ({pct:5.1f}%)")
    
    # Check for data quality issues
    print(f"\nData quality checks:")
    print(f"  NaN in X_train: {np.isnan(X_train).sum()}")
    print(f"  Inf in X_train: {np.isinf(X_train).sum()}")
    print(f"  NaN in X_test: {np.isnan(X_test).sum()}")
    print(f"  Inf in X_test: {np.isinf(X_test).sum()}")
    
    # Feature statistics
    print(f"\nFeature ranges (first 5 features):")
    for i in range(min(5, X_train.shape[1])):
        print(f"  Feature {i}: min={X_train[:, i].min():.3f}, max={X_train[:, i].max():.3f}, mean={X_train[:, i].mean():.3f}")
    
    print("="*60 + "\n")

def diagnose_training_data(X_train, X_test, y_train, y_test, feature_names):
    """
    Comprehensive diagnostic to identify data issues causing low accuracy
    """
    
    print("\n" + "="*70)
    print("CRITICAL DIAGNOSTIC - INVESTIGATING 73% ACCURACY")
    print("="*70)
    
    # 1. Dataset sizes
    print(f"\n[1] DATASET SIZES:")
    print(f"    Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"    Test:     {X_test.shape[0]} samples × {X_test.shape[1]} features")
    
    # 2. Class distribution in training set
    print(f"\n[2] CLASS DISTRIBUTION (Training):")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    total_train = len(y_train)
    
    for label, count in zip(unique_train, counts_train):
        pct = (count / total_train) * 100
        print(f"    Class {label}: {count:6d} samples ({pct:5.1f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = None
    if len(counts_train) > 1:
        imbalance_ratio = counts_train.max() / counts_train.min()
        print(f"\n    ⚠️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 5:
            print(f"    ❌ SEVERE IMBALANCE DETECTED!")
            print(f"       Model will likely predict majority class only")
    
    # 3. Class distribution in test set
    print(f"\n[3] CLASS DISTRIBUTION (Test):")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    total_test = len(y_test)
    
    for label, count in zip(unique_test, counts_test):
        pct = (count / total_test) * 100
        print(f"    Class {label}: {count:6d} samples ({pct:5.1f}%)")
    
    # Check if distributions match
    train_pcts = counts_train / total_train
    test_pcts = counts_test / total_test
    distribution_diff = np.abs(train_pcts - test_pcts).max()
    
    if distribution_diff > 0.1:
        print(f"\n    ⚠️  Train/Test Distribution Mismatch: {distribution_diff*100:.1f}%")
        print(f"       Stratified split may have failed")
    
    # 4. Data quality checks
    print(f"\n[4] DATA QUALITY:")
    
    nan_train = np.isnan(X_train).sum()
    inf_train = np.isinf(X_train).sum()
    nan_test = np.isnan(X_test).sum()
    inf_test = np.isinf(X_test).sum()
    
    print(f"    Training set:")
    print(f"      NaN values: {nan_train}")
    print(f"      Inf values: {inf_train}")
    
    print(f"    Test set:")
    print(f"      NaN values: {nan_test}")
    print(f"      Inf values: {inf_test}")
    
    if nan_train > 0 or inf_train > 0 or nan_test > 0 or inf_test > 0:
        print(f"\n    ❌ BAD DATA DETECTED! Clean this before training")
    
    # 5. Feature scaling verification
    print(f"\n[5] FEATURE SCALING (should be normalized):")
    
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    
    print(f"    Overall mean: {means.mean():.6f} (should be ~0.0)")
    print(f"    Overall std:  {stds.mean():.6f} (should be ~1.0)")
    
    if abs(means.mean()) > 0.5 or abs(stds.mean() - 1.0) > 0.5:
        print(f"\n    ⚠️  Features may not be properly scaled!")
    
    # 6. Feature ranges (first 5 features)
    print(f"\n[6] FEATURE RANGES (first 5):")
    for i in range(min(5, X_train.shape[1])):
        feat_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
        min_val = X_train[:, i].min()
        max_val = X_train[:, i].max()
        mean_val = X_train[:, i].mean()
        print(f"    {feat_name:20s}: [{min_val:8.3f}, {max_val:8.3f}] mean={mean_val:8.3f}")
    
    # 7. Check for constant features
    print(f"\n[7] FEATURE VARIANCE:")
    variances = X_train.var(axis=0)
    constant_features = np.where(variances < 1e-10)[0]
    
    if len(constant_features) > 0:
        print(f"    ❌ Found {len(constant_features)} constant/near-constant features:")
        for idx in constant_features:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"       {feat_name} (variance: {variances[idx]:.2e})")
    else:
        print(f"    ✓ All features have variance")
    
    # 8. Sample predictions from random baseline
    print(f"\n[8] BASELINE SANITY CHECK:")
    
    # Majority class baseline
    majority_class = unique_train[np.argmax(counts_train)]
    majority_baseline = np.sum(y_test == majority_class) / len(y_test)
    print(f"    Majority class baseline: {majority_baseline:.4f} ({majority_baseline*100:.2f}%)")
    print(f"    (Always predicting class {majority_class})")
    
    # Random baseline
    random_baseline = 1.0 / len(unique_train)
    print(f"    Random guess baseline:   {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    
    print(f"\n    If your model is near the majority baseline, it's likely predicting the majority class.")
    
    # 9. Recommendations
    print(f"\n" + "="*70)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues_found = []
    
    if imbalance_ratio is not None and imbalance_ratio > 5:
        issues_found.append("SEVERE CLASS IMBALANCE")
        print(f"\n❌ Issue: Severe class imbalance ({imbalance_ratio:.1f}:1)")
        print(f"   Fix: Use SMOTE or class_weight='balanced'")
    
    if nan_train > 0 or inf_train > 0 or nan_test > 0 or inf_test > 0:
        issues_found.append("BAD DATA (NaN/Inf)")
        print(f"\n❌ Issue: NaN or Inf values in data")
        print(f"   Fix: Clean data preprocessing")
    
    if abs(means.mean()) > 0.5 or abs(stds.mean() - 1.0) > 0.5:
        issues_found.append("IMPROPER SCALING")
        print(f"\n❌ Issue: Features not properly scaled")
        print(f"   Fix: Verify StandardScaler is applied")
    
    if len(constant_features) > 0:
        issues_found.append("CONSTANT FEATURES")
        print(f"\n❌ Issue: Constant features detected")
        print(f"   Fix: Remove zero-variance features")
    
    if distribution_diff > 0.1:
        issues_found.append("TRAIN/TEST MISMATCH")
        print(f"\n❌ Issue: Train/test distribution mismatch")
        print(f"   Fix: Use stratified split")
    
    if not issues_found:
        print(f"\n✓ No obvious data issues found")
        print(f"  Problem may be in model configuration or feature quality")
    
    print(f"\n" + "="*70 + "\n")
    
    return issues_found

def clean_training_data(X, y):
    """
    Remove samples with NaN or Inf values
    """
    
    print("\n=== CLEANING DATA ===")
    
    initial_samples = len(X)
    
    # Find bad rows
    nan_mask = np.isnan(X).any(axis=1)
    inf_mask = np.isinf(X).any(axis=1)
    bad_mask = nan_mask | inf_mask
    
    bad_count = bad_mask.sum()
    
    if bad_count > 0:
        print(f"Removing {bad_count} samples with NaN/Inf values")
        
        # Keep only good samples
        X_clean = X[~bad_mask]
        y_clean = y[~bad_mask]
        
        print(f"Samples: {initial_samples} → {len(X_clean)}")
        
        return X_clean, y_clean
    
    print("No bad data found")
    return X, y

def handle_class_imbalance(X_train, y_train):
    """
    Force class balance to equal class counts using SMOTE oversampling only.
    """
    
    from imblearn.over_sampling import SMOTE
    
    print("\n=== HANDLING CLASS IMBALANCE (FORCING EQUAL CLASSES) ===")
    
    # Check current distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Before balancing:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count}")
    
    try:
        # Choose k_neighbors safely relative to smallest class
        k_neighbors = int(max(1, min(5, counts.min() - 1)))
        # Oversample all classes except the majority up to the majority size
        smote = SMOTE(
            sampling_strategy='not majority',
            random_state=42,
            k_neighbors=k_neighbors
        )
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        # Show new distribution
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        print(f"\nAfter balancing (equalized to majority count):")
        for label, count in zip(unique_new, counts_new):
            print(f"  Class {label}: {count}")
        new_imbalance = counts_new.max() / counts_new.min() if len(counts_new) > 1 else 1.0
        print(f"\nNew imbalance ratio: {new_imbalance:.1f}:1")
        
        return X_balanced, y_balanced
    except Exception as e:
        print(f"⚠️  SMOTE failed: {e}")
        print(f"   Proceeding without resampling (using original data)")
        return X_train, y_train
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compare_models(X_train, X_test, y_train, y_test):
    """Compare RandomForest vs XGBoost performance"""
    
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    print("\n=== MODEL COMPARISON ===")
    
    # Train RandomForest
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # Compare
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Improvement'}")
    print("-" * 50)
    print(f"{'RandomForest':<20} {rf_acc:.4f}      Baseline")
    print(f"{'XGBoost':<20} {xgb_acc:.4f}      {(xgb_acc - rf_acc)*100:+.2f}%")
    
    return rf, xgb, rf_acc, xgb_acc

def train_individual_models(X_train, y_train, X_val, y_val):
    """Train individual models and evaluate their performance"""
    
    # Force equal class counts
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\n=== TRAINING INDIVIDUAL MODELS ===")
    
    # Calculate class weights for balanced training
    classes = np.unique(y_train_bal)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_bal
    )
    weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([weight_dict[label] for label in y_train_bal])
    
    models = {}
    
    # 1. Random Forest
    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train_bal, y_train_bal)
    rf_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)
    models['RandomForest'] = (rf, rf_acc)
    print(f"   ✓ RandomForest trained - Validation accuracy: {rf_acc:.4f}")
    
    # 2. XGBoost
    print("\n[2/3] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
        tree_method='hist',
        n_jobs=-1,
        verbosity=0
    )
    
    xgb.fit(
        X_train_bal, y_train_bal,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    xgb_pred = xgb.predict(X_val)
    xgb_acc = accuracy_score(y_val, xgb_pred)
    models['XGBoost'] = (xgb, xgb_acc)
    print(f"   ✓ XGBoost trained - Validation accuracy: {xgb_acc:.4f}")
    
    # 3. LightGBM
    print("\n[3/3] Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=250,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    from lightgbm import early_stopping as lgb_early_stopping
    lgbm.fit(
        X_train_bal, y_train_bal,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb_early_stopping(stopping_rounds=20, verbose=False)]
    )
    lgbm_pred = lgbm.predict(X_val)
    lgbm_acc = accuracy_score(y_val, lgbm_pred)
    models['LightGBM'] = (lgbm, lgbm_acc)
    print(f"   ✓ LightGBM trained - Validation accuracy: {lgbm_acc:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL PERFORMANCE")
    print(f"{'='*60}")
    for name, (model, acc) in models.items():
        print(f"{name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return models, (X_train_bal, y_train_bal)


def create_ensemble_model(models):
    """Create voting ensemble from trained models"""
    
    from sklearn.ensemble import VotingClassifier
    
    print("=== CREATING ENSEMBLE MODEL ===")
    
    # Extract models
    rf_model = models['RandomForest'][0]
    xgb_model = models['XGBoost'][0]
    lgbm_model = models['LightGBM'][0]
    
    # Create ensemble with soft voting (probability averaging)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        voting='soft',  # Use probability predictions
        weights=[1, 2, 1],  # Give XGBoost more weight
        n_jobs=-1
    )
    
    # Note: VotingClassifier will clone and fit base estimators in .fit()
    print("✓ Ensemble created (soft voting with weights [1, 2, 1])")
    
    return ensemble


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
        """Train ensemble model (RandomForest + XGBoost + LightGBM) with soft voting"""
        
        from sklearn.model_selection import train_test_split
        
        print("\n" + "="*60)
        print("ENSEMBLE MODEL TRAINING")
        print("="*60)
        
        # Split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            stratify=y_train,
            random_state=42
        )
        
        print(f"\nData split:")
        print(f"  Training: {X_tr.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        
        # Train individual models
        models, (X_tr_bal, y_tr_bal) = train_individual_models(X_tr, y_tr, X_val, y_val)
        
        # Create ensemble
        ensemble = create_ensemble_model(models)
        
        # Finalize ensemble (fit clones of base estimators) on balanced data
        print("\nFinalizing ensemble...")
        start_time = datetime.now()
        ensemble.fit(X_tr_bal, y_tr_bal)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✓ Ensemble fitted in {training_time:.2f}s")
        
        # Validate ensemble performance
        from sklearn.metrics import accuracy_score
        ensemble_val_pred = ensemble.predict(X_val)
        ensemble_val_acc = accuracy_score(y_val, ensemble_val_pred)
        
        print(f"\n{'='*60}")
        print("ENSEMBLE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Validation Accuracy: {ensemble_val_acc:.4f} ({ensemble_val_acc*100:.2f}%)")
        
        # Compare to individual models
        best_individual = max(models.values(), key=lambda x: x[1])
        improvement = ensemble_val_acc - best_individual[1]
        print(f"Best Individual:     {best_individual[1]:.4f}")
        print(f"Improvement:         {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"{'='*60}\n")
        
        # Attach details for metadata
        ensemble.individual_models_ = models
        ensemble.ensemble_type_ = 'VotingClassifier'
        
        self.classifier = ensemble
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
        
        # Feature importance (handle ensembles gracefully)
        feature_importance_dict = {}
        try:
            # Direct feature importances
            if hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                feature_importance_dict = dict(zip(self.feature_names, importances))
            # Voting ensemble: weighted average of available importances
            elif hasattr(self.classifier, 'estimators_'):
                import numpy as _np
                ests = self.classifier.estimators_
                weights = getattr(self.classifier, 'weights', None)
                if weights is None:
                    weights = [1.0] * len(ests)
                agg = _np.zeros(len(self.feature_names), dtype=float)
                total_w = 0.0
                for w, est in zip(weights, ests):
                    if hasattr(est, 'feature_importances_'):
                        vals = est.feature_importances_
                        if len(vals) == len(self.feature_names):
                            agg += w * _np.array(vals, dtype=float)
                            total_w += w
                if total_w > 0:
                    agg = agg / total_w
                    feature_importance_dict = dict(zip(self.feature_names, agg.tolist()))
        except Exception as _e:
            logger.warning(f"Feature importance unavailable: {_e}")
        
        if feature_importance_dict:
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
            'model_type': type(self.classifier).__name__,
            'training_timestamp': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'label_classes': self.label_encoder.classes_.tolist(),
            'data_shape': self.df.shape,
            'model_parameters': self.classifier.get_params(),
            'evaluation_results': self.evaluation_results,
            'version': '1.0.0'
        }
        
        # If ensemble, enrich metadata
        try:
            from sklearn.ensemble import VotingClassifier as _Voting
            if isinstance(self.classifier, _Voting) or getattr(self.classifier, 'ensemble_type_', '') == 'VotingClassifier':
                self.metadata['model_type'] = 'VotingEnsemble'
                # Composition details
                estimators = getattr(self.classifier, 'estimators', [])
                weights = getattr(self.classifier, 'weights', None) or [1]*len(estimators)
                comp = {}
                for (name, est), w in zip(estimators, weights):
                    params = {}
                    if hasattr(est, 'get_params'):
                        p = est.get_params()
                        # Keep a few salient parameters
                        for key in ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves']:
                            if key in p:
                                params[key] = p[key]
                    comp[name] = {**params, 'weight': w}
                self.metadata['ensemble_composition'] = comp
                self.metadata['voting'] = getattr(self.classifier, 'voting', 'soft')
                # Individual accuracies if available
                if hasattr(self.classifier, 'individual_models_'):
                    self.metadata['individual_accuracies'] = {
                        name: float(acc) for name, (_, acc) in self.classifier.individual_models_.items()
                    }
        except Exception as _e:
            logger.warning(f"Could not enrich ensemble metadata: {_e}")
        
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
        
        # Clean data (remove any NaN/Inf rows)
        X_train_scaled, y_train = clean_training_data(X_train_scaled, y_train)
        X_test_scaled, y_test = clean_training_data(X_test_scaled, y_test)
        
        # Comprehensive diagnostic
        issues = diagnose_training_data(X_train_scaled, X_test_scaled, y_train, y_test, trainer.feature_names)
        
        if "BAD DATA (NaN/Inf)" in issues:
            print("\n❌ STOPPING: Clean data before training")
            return
        
        # Optional quick diagnostic
        diagnose_data(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Compare models before selecting final one (optional)
        compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Train model (ensemble)
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