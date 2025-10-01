import React, { useState, useRef, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
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

// Advanced optional quality/physical fields
const ADVANCED_FIELDS = [
  { key: 'pl_snr', label: 'Transit Signal-to-Noise (SNR)', unit: '', tooltip: 'Transit model SNR; higher is better', placeholder: 'e.g., 35' },
  { key: 'koi_score', label: 'Disposition Score (0-1)', unit: '', tooltip: 'Pipeline confidence score (Kepler), 0 to 1', placeholder: 'e.g., 0.9' },
  { key: 'fp_flag_nt', label: 'Not Transit-Like Flag (0/1)', unit: '', tooltip: '1 if not transit-like, else 0', placeholder: '0 or 1' },
  { key: 'fp_flag_ss', label: 'Stellar Eclipse Flag (0/1)', unit: '', tooltip: '1 if stellar eclipse likely, else 0', placeholder: '0 or 1' },
  { key: 'fp_flag_co', label: 'Centroid Offset Flag (0/1)', unit: '', tooltip: '1 if centroid offset suggests FP, else 0', placeholder: '0 or 1' },
  { key: 'fp_flag_ec', label: 'Ephemeris Match Flag (0/1)', unit: '', tooltip: '1 if ephemeris match contamination, else 0', placeholder: '0 or 1' },
  { key: 'pl_impact', label: 'Impact Parameter', unit: '', tooltip: 'Transit impact parameter', placeholder: 'e.g., 0.5' },
  { key: 'st_slogg', label: 'Stellar log(g)', unit: '', tooltip: 'Stellar surface gravity log10(cm/s^2)', placeholder: 'e.g., 4.4' },
  { key: 'pl_eqt', label: 'Equilibrium Temperature (K)', unit: 'K', tooltip: 'Estimated equilibrium temperature', placeholder: 'e.g., 800' },
  { key: 'pl_insol', label: 'Insolation Flux (Earth flux)', unit: '', tooltip: 'Stellar flux received', placeholder: 'e.g., 100' },
  { key: 'transit_count', label: 'Number of Transits', unit: '', tooltip: 'Number of observed transits', placeholder: 'e.g., 3' }
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

// Feature Importance Chart Component
const FeatureImportanceChart = ({ modelInfo }) => {
  if (!modelInfo?.feature_importance_ranking) return null;

  // Get top 10 features and format for visualization
  const topFeatures = modelInfo.feature_importance_ranking
    .slice(0, 10)
    .map(item => ({
      name: item.feature.replace('pl_', '').replace('st_', '').replace('fp_', '').replace('_', ' '),
      importance: (item.importance * 100).toFixed(2),
      fullName: item.feature
    }));

  const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#ffeaa7', '#fab1a0', '#fd79a8'];

  return (
    <div className="card" style={{ marginTop: '30px' }}>
      <h2>🔍 Feature Importance Analysis</h2>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Which factors matter most for exoplanet classification?
      </p>
      
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={topFeatures} layout="vertical" margin={{ left: 120, right: 30, top: 20, bottom: 20 }}>
          <XAxis type="number" label={{ value: 'Importance (%)', position: 'bottom' }} />
          <YAxis type="category" dataKey="name" width={100} />
          <Tooltip 
            formatter={(value, name, props) => [
              `${value}%`, 
              `Feature: ${props.payload.fullName}`
            ]}
            contentStyle={{ background: '#fff', border: '1px solid #ddd', borderRadius: '8px' }}
          />
          <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
            {topFeatures.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      <div style={{ marginTop: '20px', fontSize: '14px', color: '#666', background: '#f8f9fa', padding: '15px', borderRadius: '8px' }}>
        <p><strong>💡 Key Insights:</strong></p>
        <ul style={{ marginLeft: '20px', marginTop: '10px' }}>
          <li>Higher values = more influential in classification decisions</li>
          <li>🔥 Quality indicators (flags, SNR) typically rank highest</li>
          <li>⚗️ Engineered features (fp_flag_any) often outperform raw data</li>
          <li>🌍 Physical parameters provide important context</li>
        </ul>
      </div>
    </div>
  );
};

// Model Performance Metrics Component
const ModelPerformanceCard = ({ modelInfo }) => {
  if (!modelInfo) return null;

  const accuracy = modelInfo.performance?.test_accuracy || modelInfo.test_accuracy || 0;
  const cvMean = modelInfo.performance?.cv_mean || modelInfo.cv_accuracy_mean || 0;
  const cvStd = modelInfo.performance?.cv_std || modelInfo.cv_accuracy_std || 0;
  const f1Score = modelInfo.f1_weighted || 0;
  const classReport = modelInfo.classification_report;
  
  // Performance level assessment
  const getPerformanceLevel = (acc) => {
    if (acc >= 0.95) return { level: 'Outstanding', color: '#00b894', emoji: '🌟' };
    if (acc >= 0.92) return { level: 'Excellent', color: '#0984e3', emoji: '⭐' };
    if (acc >= 0.90) return { level: 'Very Good', color: '#00cec9', emoji: '✨' };
    if (acc >= 0.85) return { level: 'Good', color: '#fdcb6e', emoji: '👍' };
    return { level: 'Needs Improvement', color: '#e17055', emoji: '⚠️' };
  };

  const performance = getPerformanceLevel(accuracy);
  
  return (
    <div className="card" style={{ marginTop: '30px' }}>
      <h2>📊 Enhanced Model Performance</h2>
      
      <div className="performance-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginTop: '20px' }}>
        <div className="metric-card" style={{ 
          background: `linear-gradient(135deg, ${performance.color}22 0%, ${performance.color}11 100%)`, 
          padding: '20px', 
          borderRadius: '12px', 
          textAlign: 'center', 
          border: `2px solid ${performance.color}33`,
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{ position: 'absolute', top: '10px', right: '15px', fontSize: '24px' }}>{performance.emoji}</div>
          <div className="metric-value" style={{ fontSize: '2.5rem', fontWeight: 'bold', color: performance.color, marginBottom: '10px' }}>
            {(accuracy * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: '14px', color: '#666', fontWeight: '600' }}>
            Overall Accuracy
          </div>
          <div style={{ fontSize: '12px', color: performance.color, fontWeight: '600', marginTop: '5px' }}>
            {performance.level}
          </div>
        </div>
        
        <div className="metric-card" style={{ background: 'linear-gradient(135deg, #667eea22 0%, #764ba222 100%)', padding: '20px', borderRadius: '12px', textAlign: 'center', border: '2px solid #667eea33' }}>
          <div className="metric-value" style={{ fontSize: '2rem', fontWeight: 'bold', color: '#667eea', marginBottom: '10px' }}>
            {(cvMean * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: '14px', color: '#666', fontWeight: '600' }}>
            Cross-Validation
          </div>
          <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.7 }}>
            ± {(cvStd * 100).toFixed(1)}% (very stable)
          </div>
        </div>

        <div className="metric-card" style={{ background: 'linear-gradient(135deg, #a29bfe22 0%, #6c5ce722 100%)', padding: '20px', borderRadius: '12px', textAlign: 'center', border: '2px solid #a29bfe33' }}>
          <div className="metric-value" style={{ fontSize: '2rem', fontWeight: 'bold', color: '#a29bfe', marginBottom: '10px' }}>
            {(f1Score * 100).toFixed(1)}%
          </div>
          <div className="metric-label" style={{ fontSize: '14px', color: '#666', fontWeight: '600' }}>
            F1-Score
          </div>
          <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.7 }}>
            Balanced Performance
          </div>
        </div>
      </div>

      {classReport && (
        <div style={{ marginTop: '30px' }}>
          <h3>📈 Per-Class Performance</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '15px', marginTop: '15px' }}>
            {Object.entries(classReport).map(([className, metrics]) => {
              if (typeof metrics === 'object' && className !== 'accuracy' && metrics.support) {
                const classColors = {
                  'CONFIRMED': '#00b894',
                  'CANDIDATE': '#0984e3', 
                  'FALSE POSITIVE': '#e17055'
                };
                const color = classColors[className] || '#667eea';
                
                return (
                  <div key={className} style={{
                    background: `linear-gradient(135deg, ${color}15 0%, ${color}08 100%)`,
                    padding: '15px',
                    borderRadius: '10px',
                    border: `2px solid ${color}20`
                  }}>
                    <h4 style={{ color, margin: '0 0 10px 0', fontSize: '16px' }}>
                      {className} ({metrics.support} samples)
                    </h4>
                    <div style={{ fontSize: '13px', color: '#666' }}>
                      <div>Precision: <strong style={{color}}>{(metrics.precision * 100).toFixed(1)}%</strong></div>
                      <div>Recall: <strong style={{color}}>{(metrics.recall * 100).toFixed(1)}%</strong></div>
                      <div>F1-Score: <strong style={{color}}>{(metrics['f1-score'] * 100).toFixed(1)}%</strong></div>
                    </div>
                  </div>
                );
              }
              return null;
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// Enhanced Features Summary Component
const EnhancedFeaturesCard = ({ modelInfo }) => {
  if (!modelInfo) return null;

  const totalFeatures = modelInfo.features?.length || 0;
  const qualityFeatures = modelInfo.quality_features || [];
  const physicalFeatures = modelInfo.physical_features || [];
  
  const featureCategories = [
    { name: 'Quality Indicators', count: qualityFeatures.length, color: '#e84393', icon: '🔥' },
    { name: 'Physical Features', count: physicalFeatures.length, color: '#00b894', icon: '⭐' },
    { name: 'Other Features', count: totalFeatures - qualityFeatures.length - physicalFeatures.length, color: '#74b9ff', icon: '🌟' }
  ];

  return (
    <div className="card" style={{ marginTop: '30px' }}>
      <h2>🚀 Enhanced Feature Set</h2>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Upgraded from 8 to {totalFeatures} features with advanced quality indicators
      </p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
        {featureCategories.map(category => (
          <div key={category.name} style={{
            background: `linear-gradient(135deg, ${category.color}22 0%, ${category.color}11 100%)`,
            padding: '20px',
            borderRadius: '12px',
            textAlign: 'center',
            border: `2px solid ${category.color}33`
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '10px' }}>{category.icon}</div>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: category.color, marginBottom: '8px' }}>
              {category.count}
            </div>
            <div style={{ fontSize: '14px', color: '#666', fontWeight: '600' }}>
              {category.name}
            </div>
          </div>
        ))}
      </div>
      
      {qualityFeatures.length > 0 && (
        <div style={{ marginTop: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
          <p><strong>🔥 Critical Quality Features:</strong></p>
          <div style={{ fontSize: '14px', color: '#666', marginTop: '8px' }}>
            {qualityFeatures.slice(0, 5).join(', ')}
            {qualityFeatures.length > 5 && '...'}
          </div>
        </div>
      )}
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
  const [showAdvanced, setShowAdvanced] = useState(false);

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

        {/* Advanced Quality Section */}
        <div className="advanced-section">
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', margin: '16px 0' }}>
            <button
              type="button"
              className="toggle-advanced-button"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? 'Hide Advanced Quality Inputs' : 'Show Advanced Quality Inputs'}
            </button>
            <button
              type="button"
              className="prefill-button"
              onClick={() => {
                setShowAdvanced(true);
                setFormData(prev => ({
                  ...prev,
                  pl_snr: prev.pl_snr ?? 35,
                  koi_score: prev.koi_score ?? 0.95,
                  fp_flag_nt: prev.fp_flag_nt ?? 0,
                  fp_flag_ss: prev.fp_flag_ss ?? 0,
                  fp_flag_co: prev.fp_flag_co ?? 0,
                  fp_flag_ec: prev.fp_flag_ec ?? 0
                }));
              }}
              title="Prefill strong quality defaults (SNR 35, KOI 0.95, flags 0)"
            >
              ⭐ Prefill good-quality defaults
            </button>
          </div>

          {showAdvanced && (
            <div className="form-grid advanced-grid">
              {ADVANCED_FIELDS.map((field) => (
                <div key={field.key} className="form-group">
                  <label htmlFor={field.key} className="form-label">
                    {field.label}{field.unit ? ` (${field.unit})` : ''}
                    <span className="tooltip" title={field.tooltip}>ℹ️</span>
                  </label>
                  <input
                    type="number"
                    id={field.key}
                    step="any"
                    placeholder={field.placeholder}
                    value={formData[field.key] ?? ''}
                    onChange={(e) => handleInputChange(field.key, e.target.value)}
                    className="form-input"
                  />
                </div>
              ))}
            </div>
          )}
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
  const [modelInfo, setModelInfo] = useState(null);
  
  // Fetch model information on component mount
  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/model/info`);
        if (response.ok) {
          const info = await response.json();
          setModelInfo(info);
          console.log('Model info loaded:', info);
        }
      } catch (error) {
        console.warn('Could not fetch model info:', error);
      }
    };
    
    fetchModelInfo();
  }, []);

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
        
        {/* Model Analytics Section - Always visible when model info is loaded */}
        {modelInfo && (
          <div className="analytics-container">
            <ModelPerformanceCard modelInfo={modelInfo} />
            <EnhancedFeaturesCard modelInfo={modelInfo} />
            <FeatureImportanceChart modelInfo={modelInfo} />
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