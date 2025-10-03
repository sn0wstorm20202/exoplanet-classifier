#!/usr/bin/env python3
"""
FastAPI Backend for Exoplanet Classification

This FastAPI server provides endpoints for exoplanet classification using
a trained RandomForest model. It includes:
1. Health check endpoint
2. Model information endpoint
3. CSV file upload prediction endpoint
4. Sample prediction endpoint for testing
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    pl_orbper: Optional[float] = Field(None, description="Orbital Period [days]")
    pl_rade: Optional[float] = Field(None, description="Planet Radius [Earth radii]")
    pl_trandep: Optional[float] = Field(None, description="Transit Depth [ppm]")
    pl_trandur: Optional[float] = Field(None, description="Transit Duration [hours]")
    pl_bmasse: Optional[float] = Field(None, description="Planet Mass [Earth masses]")
    st_teff: Optional[float] = Field(None, description="Stellar Temperature [K]")
    st_rad: Optional[float] = Field(None, description="Stellar Radius [Solar radii]")
    sy_dist: Optional[float] = Field(None, description="Distance [pc]")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    processing_info: Dict[str, Any]

class ExoplanetClassifier:
    """Exoplanet classifier wrapper."""
    
    def __init__(self):
        """Initialize classifier with model components."""
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.feature_names = None
        self.is_loaded = False
        
        # Set up paths
        self.script_dir = Path(__file__).parent.parent
        self.models_dir = self.script_dir / 'models'
        
        # Check if running in Docker (models mounted at /app/models)
        if Path('/app/models').exists():
            self.models_dir = Path('/app/models')
        
    def load_models(self):
        """Load trained model components."""
        logger.info("🔄 Loading model components...")
        
        try:
            # Model files to load
            model_files = {
                'classifier.pkl': 'classifier',
                'scaler.pkl': 'scaler', 
                'label_encoder.pkl': 'label_encoder'
            }
            
            # Load model components
            for filename, attr_name in model_files.items():
                filepath = self.models_dir / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Model file not found: {filepath}")
                
                with open(filepath, 'rb') as f:
                    setattr(self, attr_name, pickle.load(f))
                logger.info(f"✅ Loaded {filename}")
            
            # Load metadata
            metadata_path = self.models_dir / 'model_metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            logger.info(f"✅ Loaded metadata: {len(self.feature_names)} features")
            
            self.is_loaded = True
            logger.info("🎉 Model components loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model components: {str(e)}")
            raise
    
    def validate_and_prepare_data(self, df: pd.DataFrame):
        """Validate and prepare data for prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        logger.info(f"📊 Input data shape: {df.shape}")
        
        # Check available features
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]
        
        logger.info(f"✅ Available features: {available_features}")
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        if not available_features:
            raise ValueError("No valid features found in input data")
        
        # Create feature matrix with all required features
        X = pd.DataFrame()
        feature_info = {}
        
        for feature in self.feature_names:
            if feature in df.columns:
                X[feature] = df[feature]
                feature_info[feature] = {
                    'available': True,
                    'missing_count': int(df[feature].isnull().sum())
                }
            else:
                # Use median from training data if available in metadata
                median_value = 0.0  # Default fallback
                if ('evaluation_results' in self.metadata and 
                    'feature_importance' in self.metadata['evaluation_results']):
                    # Use a reasonable default based on feature type
                    feature_defaults = {
                        'pl_orbper': 20.0,    # ~20 days orbital period
                        'pl_rade': 2.0,       # ~2 Earth radii
                        'pl_trandep': 1000.0, # ~1000 ppm transit depth
                        'pl_trandur': 3.0,    # ~3 hours transit duration
                        'pl_bmasse': 5.0,     # ~5 Earth masses
                        'st_teff': 5778.0,    # ~Sun temperature
                        'st_rad': 1.0,        # ~Solar radius
                        'sy_dist': 200.0      # ~200 pc distance
                    }
                    median_value = feature_defaults.get(feature, 0.0)
                
                X[feature] = median_value
                feature_info[feature] = {
                    'available': False,
                    'filled_with': float(median_value)
                }
        
        # Fill missing values with median of available data
        for feature in available_features:
            if X[feature].isnull().any():
                median_val = X[feature].median()
                X[feature] = X[feature].fillna(median_val)
                logger.info(f"📝 Filled missing values in {feature} with {median_val:.3f}")
        
        return X, feature_info
    
    def predict(self, df: pd.DataFrame):
        """Make predictions on input data."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        # Validate and prepare data
        X, feature_info = self.validate_and_prepare_data(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.classifier.predict(X_scaled)
        prediction_probabilities = self.classifier.predict_proba(X_scaled)
        
        # Convert predictions to labels
        prediction_labels = self.label_encoder.inverse_transform(predictions)
        
        # Prepare results
        results = []
        for i in range(len(df)):
            # Get probabilities for all classes
            probs = prediction_probabilities[i]
            prob_dict = {}
            confidence_scores = {}
            
            for j, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[class_name] = float(probs[j])
                confidence_scores[class_name] = float(probs[j])
            
            # Calculate confidence (probability of predicted class)
            predicted_class_idx = predictions[i]
            confidence = float(probs[predicted_class_idx])
            
            result = {
                'row_index': int(i),
                'prediction': str(prediction_labels[i]),
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'confidence_scores': confidence_scores
            }
            results.append(result)
        
        # Summary statistics
        prediction_counts = pd.Series(prediction_labels).value_counts().to_dict()
        # Convert numpy types to Python types
        prediction_counts = {str(k): int(v) for k, v in prediction_counts.items()}
        avg_confidence = float(np.mean([r['confidence'] for r in results]))
        
        summary = {
            'total_predictions': int(len(results)),
            'prediction_distribution': prediction_counts,
            'average_confidence': float(avg_confidence),
            'feature_info': feature_info
        }
        
        processing_info = {
            'model_version': self.metadata.get('version', '1.0.0'),
            'features_used': self.feature_names,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return {
            'predictions': results,
            'summary': summary,
            'processing_info': processing_info
        }

# Initialize classifier
classifier = ExoplanetClassifier()

# Create FastAPI app
app = FastAPI(
    title="Exoplanet Classification API",
    description="API for classifying exoplanets using NASA datasets and machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model components on startup."""
    try:
        classifier.load_models()
        logger.info("🚀 API server started successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {str(e)}")
        raise

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Exoplanet Classification API is running",
        "model_loaded": classifier.is_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract key information from metadata
    info = {
        "model_type": classifier.metadata.get("model_type"),
        "version": classifier.metadata.get("version"),
        "training_timestamp": classifier.metadata.get("training_timestamp"),
        "features": classifier.feature_names,
        "label_classes": classifier.metadata.get("label_classes", []),
        "data_shape": classifier.metadata.get("data_shape"),
        "performance": {
            "test_accuracy": classifier.metadata.get("evaluation_results", {}).get("test_accuracy"),
            "cv_mean": classifier.metadata.get("evaluation_results", {}).get("cv_mean"),
            "cv_std": classifier.metadata.get("evaluation_results", {}).get("cv_std")
        },
        "feature_importance": classifier.metadata.get("evaluation_results", {}).get("feature_importance", {})
    }
    
    # Ensemble-specific info
    if classifier.metadata.get("model_type") == "VotingEnsemble":
        info["ensemble_composition"] = classifier.metadata.get("ensemble_composition", {})
        info["individual_accuracies"] = classifier.metadata.get("individual_accuracies", {})
        voting = classifier.metadata.get("voting")
        if voting:
            info["voting_method"] = f"{voting} (probability averaging)"
    
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    """Predict exoplanet classifications from CSV file."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info(f"📁 Received CSV with shape: {df.shape}")
        
        # Make predictions
        results = classifier.predict(df)
        
        return PredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/sample")
async def predict_sample():
    """Test prediction with sample data."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Sample exoplanet data
    sample_data = {
        'pl_orbper': [365.25, 10.5, 88.0, 30.0, 150.0],  # Orbital period in days
        'pl_rade': [1.0, 2.5, 0.8, 3.2, 1.5],             # Planet radius in Earth radii
        'pl_trandep': [800, 1200, 600, 1500, 900],        # Transit depth in ppm
        'pl_trandur': [3.5, 2.8, 4.1, 3.0, 3.7],         # Transit duration in hours
        'pl_bmasse': [1.0, 8.0, 0.7, 15.0, 2.5],         # Planet mass in Earth masses
        'st_teff': [5778, 4500, 6200, 5000, 5900],        # Stellar temperature in K
        'st_rad': [1.0, 0.8, 1.2, 0.9, 1.1],             # Stellar radius in Solar radii
        'sy_dist': [150, 200, 100, 250, 175]              # Distance in pc
    }
    
    try:
        df = pd.DataFrame(sample_data)
        results = classifier.predict(df)
        
        return PredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"❌ Sample prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sample prediction failed: {str(e)}")

@app.post("/predict/single")
async def predict_single(request: PredictionRequest):
    """Predict classification for a single exoplanet."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        data_dict = request.dict()
        
        # Remove None values and create DataFrame
        filtered_data = {k: [v] for k, v in data_dict.items() if v is not None}
        
        if not filtered_data:
            raise HTTPException(status_code=400, detail="At least one feature must be provided")
        
        df = pd.DataFrame(filtered_data)
        
        # Make prediction
        results = classifier.predict(df)
        
        return PredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"❌ Single prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Single prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )