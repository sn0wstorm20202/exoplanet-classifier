import React, { useState, useRef } from 'react';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

// Feature field definitions
const FEATURE_FIELDS = [
  {
    key: 'pl_orbper',
    label: 'Orbital Period',
    unit: 'days',
    tooltip: 'Time for the planet to complete one orbit around its star',
    placeholder: 'e.g., 365.25'
  },
  {
    key: 'pl_rade',
    label: 'Planet Radius',
    unit: 'Earth radii',
    tooltip: 'Radius of the planet compared to Earth',
    placeholder: 'e.g., 1.0'
  },
  {
    key: 'pl_trandep',
    label: 'Transit Depth',
    unit: 'ppm',
    tooltip: 'Depth of the light curve during transit',
    placeholder: 'e.g., 1000'
  },
  {
    key: 'pl_trandur',
    label: 'Transit Duration',
    unit: 'hours',
    tooltip: 'Duration of the planet transit across the star',
    placeholder: 'e.g., 3.5'
  },
  {
    key: 'pl_bmasse',
    label: 'Planet Mass',
    unit: 'Earth masses',
    tooltip: 'Mass of the planet compared to Earth',
    placeholder: 'e.g., 1.0'
  },
  {
    key: 'st_teff',
    label: 'Stellar Temperature',
    unit: 'K',
    tooltip: 'Effective temperature of the host star',
    placeholder: 'e.g., 5778'
  },
  {
    key: 'st_rad',
    label: 'Stellar Radius',
    unit: 'Solar radii',
    tooltip: 'Radius of the host star compared to the Sun',
    placeholder: 'e.g., 1.0'
  },
  {
    key: 'sy_dist',
    label: 'System Distance',
    unit: 'pc',
    tooltip: 'Distance from Earth to the planetary system',
    placeholder: 'e.g., 200'
  }
];

// Classification badge component
const ClassificationBadge = ({ prediction, confidence }) => {
  const getBadgeClass = (pred) => {
    switch (pred) {
      case 'CONFIRMED':
        return 'badge-confirmed';
      case 'CANDIDATE':
        return 'badge-candidate';
      case 'FALSE POSITIVE':
        return 'badge-false-positive';
      default:
        return 'badge-unknown';
    }
  };

  return (
    <div className={`classification-badge ${getBadgeClass(prediction)}`}>
      <span className="badge-text">{prediction}</span>
      <span className="badge-confidence">({(confidence * 100).toFixed(1)}%)</span>
    </div>
  );
};

// Confidence bar component
const ConfidenceBar = ({ confidence, prediction }) => {
  const getBarClass = (pred) => {
    switch (pred) {
      case 'CONFIRMED':
        return 'confidence-bar-confirmed';
      case 'CANDIDATE':
        return 'confidence-bar-candidate';
      case 'FALSE POSITIVE':
        return 'confidence-bar-false-positive';
      default:
        return 'confidence-bar-unknown';
    }
  };

  return (
    <div className="confidence-bar-container">
      <div className={`confidence-bar ${getBarClass(prediction)}`}>
        <div 
          className="confidence-fill" 
          style={{ width: `${confidence * 100}%` }}
        ></div>
      </div>
      <span className="confidence-text">{(confidence * 100).toFixed(1)}%</span>
    </div>
  );
};

// Summary cards component
const SummaryCards = ({ summary, isLoading }) => {
  if (isLoading) {
    return (
      <div className="summary-cards">
        <div className="summary-card">
          <div className="summary-value loading">Loading...</div>
          <div className="summary-label">Processing</div>
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const { total_predictions, prediction_distribution, average_confidence } = summary;
  
  return (
    <div className="summary-cards">
      <div className="summary-card">
        <div className="summary-value">{total_predictions}</div>
        <div className="summary-label">Total Predictions</div>
      </div>
      <div className="summary-card confirmed">
        <div className="summary-value">{prediction_distribution?.CONFIRMED || 0}</div>
        <div className="summary-label">Confirmed</div>
      </div>
      <div className="summary-card candidate">
        <div className="summary-value">{prediction_distribution?.CANDIDATE || 0}</div>
        <div className="summary-label">Candidate</div>
      </div>
      <div className="summary-card false-positive">
        <div className="summary-value">{prediction_distribution?.['FALSE POSITIVE'] || 0}</div>
        <div className="summary-label">False Positive</div>
      </div>
      <div className="summary-card">
        <div className="summary-value">{(average_confidence * 100).toFixed(1)}%</div>
        <div className="summary-label">Avg Confidence</div>
      </div>
    </div>
  );
};

// Results table component
const ResultsTable = ({ predictions, isLoading }) => {
  if (isLoading) {
    return (
      <div className="results-section">
        <h3>Classification Results</h3>
        <div className="loading-message">Processing your data...</div>
      </div>
    );
  }

  if (!predictions || predictions.length === 0) {
    return null;
  }

  return (
    <div className="results-section">
      <h3>Classification Results</h3>
      <div className="results-table-container">
        <table className="results-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Probabilities</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((result, index) => (
              <tr key={index}>
                <td>{result.row_index + 1}</td>
                <td>
                  <ClassificationBadge 
                    prediction={result.prediction} 
                    confidence={result.confidence} 
                  />
                </td>
                <td>
                  <ConfidenceBar 
                    confidence={result.confidence} 
                    prediction={result.prediction} 
                  />
                </td>
                <td>
                  <div className="probabilities">
                    {Object.entries(result.probabilities).map(([label, prob]) => (
                      <span key={label} className="probability">
                        {label}: {(prob * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Single planet analysis tab
const SinglePlanetTab = ({ onResults, isLoading }) => {
  const [formData, setFormData] = useState({});

  const handleInputChange = (key, value) => {
    setFormData(prev => ({
      ...prev,
      [key]: value === '' ? null : parseFloat(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Filter out empty values
    const filteredData = Object.fromEntries(
      Object.entries(formData).filter(([_, value]) => value !== null && value !== undefined && value !== '')
    );

    if (Object.keys(filteredData).length === 0) {
      alert('Please enter at least one feature value.');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict/single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filteredData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      onResults(results);
    } catch (error) {
      console.error('Prediction error:', error);
      alert(`Prediction failed: ${error.message}`);
    }
  };

  return (
    <div className="tab-content">
      <h2>Single Planet Analysis</h2>
      <p className="tab-description">
        Enter the parameters of an exoplanet to get its classification. 
        You don't need to fill all fields - the model will use available data.
      </p>
      
      <form onSubmit={handleSubmit} className="single-planet-form">
        <div className="form-grid">
          {FEATURE_FIELDS.map((field) => (
            <div key={field.key} className="form-group">
              <label htmlFor={field.key} className="form-label">
                {field.label} ({field.unit})
                <span className="tooltip" title={field.tooltip}>ℹ️</span>
              </label>
              <input
                type="number"
                id={field.key}
                step="any"
                placeholder={field.placeholder}
                value={formData[field.key] || ''}
                onChange={(e) => handleInputChange(field.key, e.target.value)}
                className="form-input"
              />
            </div>
          ))}
        </div>
        
        <button 
          type="submit" 
          className="classify-button"
          disabled={isLoading}
        >
          {isLoading ? 'Classifying...' : '🚀 Classify Exoplanet'}
        </button>
      </form>
    </div>
  );
};

// Batch analysis tab
const BatchAnalysisTab = ({ onResults, isLoading }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (!file.name.endsWith('.csv')) {
      alert('Please select a CSV file.');
      return;
    }
    setSelectedFile(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
      alert('Please select a CSV file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      onResults(results);
    } catch (error) {
      console.error('Batch prediction error:', error);
      alert(`Batch prediction failed: ${error.message}`);
    }
  };

  const testWithSample = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict/sample`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const results = await response.json();
      onResults(results);
    } catch (error) {
      console.error('Sample prediction error:', error);
      alert(`Sample prediction failed: ${error.message}`);
    }
  };

  return (
    <div className="tab-content">
      <h2>Batch Analysis</h2>
      <p className="tab-description">
        Upload a CSV file with exoplanet data for batch classification. 
        The CSV should include columns matching the feature names.
      </p>

      <form onSubmit={handleSubmit}>
        <div 
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'file-selected' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            accept=".csv"
            onChange={handleFileInputChange}
            className="file-input-hidden"
          />
          
          {selectedFile ? (
            <div className="file-info">
              <div className="file-icon">📄</div>
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-size">{(selectedFile.size / 1024).toFixed(1)} KB</div>
            </div>
          ) : (
            <div className="drop-zone-content">
              <div className="drop-icon">📁</div>
              <div className="drop-text">
                Drag and drop your CSV file here, or <span className="click-text">click to browse</span>
              </div>
              <div className="drop-subtext">CSV files only, max 10MB</div>
            </div>
          )}
        </div>

        <div className="button-group">
          <button 
            type="submit" 
            className="analyze-button"
            disabled={!selectedFile || isLoading}
          >
            {isLoading ? 'Analyzing...' : '📊 Analyze Batch'}
          </button>
          
          <button 
            type="button" 
            className="sample-button"
            onClick={testWithSample}
            disabled={isLoading}
          >
            {isLoading ? 'Testing...' : '🧪 Test with Sample Data'}
          </button>
        </div>
      </form>

      <div className="csv-format-info">
        <h4>Expected CSV Format</h4>
        <p>Your CSV file should include columns with these names:</p>
        <div className="csv-columns">
          {FEATURE_FIELDS.map((field, index) => (
            <span key={field.key} className="csv-column">
              {field.key}
              {index < FEATURE_FIELDS.length - 1 ? ', ' : ''}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

// Main App component
const App = () => {
  const [activeTab, setActiveTab] = useState('single');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleResults = (newResults) => {
    setResults(newResults);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResults(null); // Clear results when switching tabs
  };

  // Wrapper functions to handle loading state
  const handleSinglePlanetResults = async (resultsPromise) => {
    setIsLoading(true);
    try {
      await resultsPromise;
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchResults = async (resultsPromise) => {
    setIsLoading(true);
    try {
      await resultsPromise;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🌌 Exoplanet Classifier</h1>
        <p>AI-powered exoplanet classification using NASA datasets</p>
      </header>

      <nav className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => handleTabChange('single')}
        >
          🪐 Single Planet Analysis
        </button>
        <button 
          className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => handleTabChange('batch')}
        >
          📊 Batch Analysis
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'single' ? (
          <SinglePlanetTab 
            onResults={handleResults} 
            isLoading={isLoading}
          />
        ) : (
          <BatchAnalysisTab 
            onResults={handleResults} 
            isLoading={isLoading}
          />
        )}

        {(results || isLoading) && (
          <div className="results-container">
            <SummaryCards 
              summary={results?.summary} 
              isLoading={isLoading}
            />
            <ResultsTable 
              predictions={results?.predictions} 
              isLoading={isLoading}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Built with NASA exoplanet data • RandomForest ML model • React + FastAPI</p>
      </footer>
    </div>
  );
};

export default App;