# Exoplanet Classifier Frontend

React frontend for the NASA Exoplanet Classification system built with Vite.

## 🚀 Deployment Guide

### Vercel Deployment (Recommended)

1. **Push to GitHub**: Ensure your frontend code is in a GitHub repository

2. **Connect to Vercel**:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Select the `frontend` folder as the root directory

3. **Configure Environment Variables**:
   - In Vercel project settings, go to "Environment Variables"
   - Add the following variables:
     ```
     VITE_API_BASE_URL = https://your-backend-url.onrender.com
     VITE_APP_ENV = production
     VITE_APP_NAME = Exoplanet Classifier
     VITE_APP_VERSION = 1.0.0
     ```

4. **Deploy**: Vercel will automatically deploy on every push to main branch

### Manual Deployment

```bash
# Install dependencies
npm install

# Build for production
npm run build

# Deploy the dist/ folder to your hosting service
```

## 🔧 Environment Configuration

### Development (.env.development)
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_ENV=development
```

### Production (.env.production)
```env
VITE_API_BASE_URL=https://your-backend-url.onrender.com
VITE_APP_ENV=production
```

## 📝 Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## 🔗 Backend Integration

The frontend expects the following API endpoints:

- `GET /` - Health check
- `GET /model/info` - Model metadata
- `POST /predict/single` - Single planet prediction
- `POST /predict` - Batch CSV prediction
- `POST /predict/sample` - Sample data test

## 🌐 CORS Configuration

Ensure your backend (Render deployment) includes the following CORS settings:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-vercel-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚨 Important Notes

1. **Update API URL**: After deploying backend to Render, update `VITE_API_BASE_URL` in `.env.production`
2. **Environment Variables**: Vercel automatically loads `.env.production` for production builds
3. **CORS**: Ensure your backend allows requests from your Vercel domain
4. **Build Optimization**: The build is optimized for production with code splitting and minification

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failed**:
   - Check if `VITE_API_BASE_URL` is correctly set
   - Verify backend CORS settings
   - Check browser network tab for CORS errors

2. **Environment Variables Not Loading**:
   - Ensure variables are prefixed with `VITE_`
   - Check Vercel environment variable settings
   - Rebuild and redeploy

3. **Build Fails**:
   - Check Node.js version (>=18.0.0 required)
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check for TypeScript/lint errors

## 📦 Dependencies

- **React 18.2.0** - UI framework
- **Vite 5.0.8** - Build tool and dev server
- **Recharts 3.2.1** - Data visualization charts

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── dist/                # Production build output
├── .env.development     # Development environment
├── .env.production      # Production environment
├── vercel.json          # Vercel deployment config
└── vite.config.js       # Vite configuration
```