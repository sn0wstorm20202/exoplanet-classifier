# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a complete web application for classifying exoplanets using NASA datasets and machine learning. It combines three NASA exoplanet datasets (K2, Kepler, and TESS) to train a RandomForest classifier that predicts whether a celestial object is a **Confirmed Exoplanet**, **Planet Candidate**, or **False Positive**.

The project achieves 85-95% accuracy using 8 key astronomical features and provides both single planet analysis and batch CSV processing through a modern React frontend with FastAPI backend.

## Development Commands

### Data Processing & Model Training
```powershell
# Process NASA datasets (required first step)
python notebooks/load_combined_data.py

# Train the RandomForest model
python backend/train.py
```

### Backend Development
```powershell
# Install Python dependencies
pip install -r backend/requirements.txt

# Start FastAPI development server
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Alternative: run directly
python backend/app.py
```

### Frontend Development
```powershell
# Install Node.js dependencies
cd frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Serve production build
npm run serve

# Lint code
npm run lint
```

### Docker Deployment
```powershell
# Full deployment (recommended for first-time setup)
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild specific service
docker-compose build backend
docker-compose build frontend
```

### Testing & Validation
```powershell
# Test API health
curl http://localhost:8000

# Test model info
curl http://localhost:8000/model/info

# Test sample prediction
curl -X POST http://localhost:8000/predict/sample

# Test single prediction
curl -X POST http://localhost:8000/predict/single -H "Content-Type: application/json" -d '{"pl_orbper": 365.25, "pl_rade": 1.0, "st_teff": 5778}'
```

## Architecture Overview

### Data Flow Pipeline
The system follows a structured pipeline:
1. **Data Ingestion**: `notebooks/load_combined_data.py` processes NASA CSV files
2. **Feature Mapping**: Standardizes different column name formats across datasets
3. **Data Processing**: Handles missing values, removes outliers, standardizes labels
4. **Model Training**: `backend/train.py` trains RandomForest with StandardScaler
5. **Model Serving**: `backend/app.py` provides REST API endpoints
6. **Web Interface**: React frontend with dual analysis modes

### Backend Architecture (FastAPI)
- **ExoplanetClassifier**: Core model wrapper handling prediction logic
- **Model Loading**: Automatic loading of pickled classifier, scaler, and label encoder
- **Data Validation**: Comprehensive input validation and feature mapping
- **Error Handling**: Robust error handling with meaningful HTTP responses
- **CORS Support**: Configured for frontend-backend communication

Key endpoints:
- `GET /`: Health check and model status
- `GET /model/info`: Model metadata and performance metrics
- `POST /predict`: CSV batch processing
- `POST /predict/single`: Single planet analysis
- `POST /predict/sample`: Test with sample data

### Frontend Architecture (React + Vite)
- **Single Page Application**: React with modern hooks pattern
- **Dual Analysis Modes**: Toggle between single planet input and CSV batch upload
- **Real-time Results**: Dynamic classification badges and confidence visualization
- **Responsive Design**: Purple gradient theme with mobile-friendly interface
- **API Integration**: Fetch-based communication with backend

### Machine Learning Pipeline
- **Features**: 8 standardized astronomical parameters (orbital period, radius, mass, etc.)
- **Algorithm**: RandomForest (200 estimators, depth 15, balanced weights)
- **Preprocessing**: StandardScaler for feature normalization
- **Labels**: Three-class classification (CONFIRMED/CANDIDATE/FALSE POSITIVE)
- **Evaluation**: Cross-validation, accuracy metrics, feature importance analysis

### Data Management
- **Raw Data**: NASA CSV files in `data/` (k2_data.csv, kepler_data.csv, tess_data.csv)
- **Processed Data**: `data/processed_combined.csv` (combined and standardized)
- **Model Artifacts**: `models/` directory (classifier.pkl, scaler.pkl, label_encoder.pkl, metadata.json)
- **Feature Mapping**: Flexible column name mapping for different NASA dataset formats

### Docker Architecture
- **Multi-service Setup**: Backend, frontend, and model-init containers
- **Model Initialization**: Dedicated container for data processing and training
- **Volume Mounting**: Shared models and data directories
- **Health Checks**: Automated service health monitoring
- **Dependency Management**: Proper service startup ordering

## Key Technical Details

### Feature Processing
The system handles diverse NASA dataset formats by mapping columns like:
- Orbital Period: `pl_orbper`, `Orbital Period [days]`, `koi_period`
- Planet Radius: `pl_rade`, `Planetary Radius [Earth radii]`, `koi_prad`
- Label Columns: `Disposition Using Kepler Data`, `koi_disposition`, `tfopwg_disp`

Missing features are filled with reasonable defaults, and outliers beyond 3 standard deviations are removed.

### Model Metadata
The system maintains comprehensive metadata including:
- Training timestamp and data shape
- Feature importance rankings
- Cross-validation scores and confusion matrix
- Model hyperparameters and performance metrics

### Error Handling
- Missing model files trigger clear error messages
- Invalid CSV formats are rejected with helpful feedback
- Feature validation ensures required data is present
- Graceful degradation for missing features using defaults

### Development Workflow
1. Ensure NASA CSV files are in `data/` directory
2. Run data processing to create `processed_combined.csv`
3. Train model to generate artifacts in `models/`
4. Start backend API server
5. Launch frontend development server
6. Test endpoints and functionality

The Docker setup automates this entire workflow, making it ideal for deployment and testing.