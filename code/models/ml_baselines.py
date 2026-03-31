"""
Traditional machine learning baselines for SSC estimation.
Includes SVM, LightGBM, XGBoost and empirical indices.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings

# Try to import optional dependencies
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


class SklearnWrapper:
    """
    Wrapper to make sklearn models compatible with PyTorch training pipeline.
    """
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.device = 'cpu'
    
    def fit(self, X, y):
        """Fit the model."""
        # Flatten image data if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Predict using the model."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def __call__(self, X):
        """Make callable for PyTorch compatibility."""
        # Convert torch tensor to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        predictions = self.predict(X)
        return torch.tensor(predictions, dtype=torch.float32)


class SVMBaseline:
    """
    Support Vector Machine baseline for SSC estimation.
    Can use either RGB bands only or full spectral bands.
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', use_rgb_only=False):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        self.wrapper = None
        self.use_rgb_only = use_rgb_only
    
    def prepare_data(self, images):
        """
        Prepare image data for SVM.
        Args:
            images: (N, C, H, W) or (N, H, W, C) array
        Returns:
            features: (N, features) flattened array
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Select RGB bands only if specified
        if self.use_rgb_only and images.shape[1] >= 3:
            images = images[:, :3, :, :]  # First 3 bands
        
        # Flatten spatial dimensions
        N = images.shape[0]
        features = images.reshape(N, -1)
        return features
    
    def fit(self, images, labels):
        """Train SVM model."""
        features = self.prepare_data(images)
        self.wrapper = SklearnWrapper(self.model)
        self.wrapper.fit(features, labels)
        return self
    
    def predict(self, images):
        """Predict SSC values."""
        if self.wrapper is None:
            raise RuntimeError("Model not fitted yet")
        features = self.prepare_data(images)
        return self.wrapper.predict(features)


class LightGBMBaseline:
    """
    LightGBM baseline for SSC estimation.
    Efficient gradient boosting framework.
    """
    def __init__(self, use_rgb_only=False, **kwargs):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 100
        }
        default_params.update(kwargs)
        
        self.model = lgb.LGBMRegressor(**default_params)
        self.wrapper = None
        self.use_rgb_only = use_rgb_only
    
    def prepare_data(self, images):
        """Prepare image data."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if self.use_rgb_only and images.shape[1] >= 3:
            images = images[:, :3, :, :]
        
        N = images.shape[0]
        return images.reshape(N, -1)
    
    def fit(self, images, labels, eval_set=None):
        """Train LightGBM model."""
        features = self.prepare_data(images)
        self.wrapper = SklearnWrapper(self.model)
        self.wrapper.fit(features, labels)
        return self
    
    def predict(self, images):
        """Predict SSC values."""
        if self.wrapper is None:
            raise RuntimeError("Model not fitted yet")
        features = self.prepare_data(images)
        return self.wrapper.predict(features)


class XGBoostBaseline:
    """
    XGBoost baseline for SSC estimation.
    Extreme Gradient Boosting with regularization.
    """
    def __init__(self, use_rgb_only=False, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.wrapper = None
        self.use_rgb_only = use_rgb_only
    
    def prepare_data(self, images):
        """Prepare image data."""
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        if self.use_rgb_only and images.shape[1] >= 3:
            images = images[:, :3, :, :]
        
        N = images.shape[0]
        return images.reshape(N, -1)
    
    def fit(self, images, labels):
        """Train XGBoost model."""
        features = self.prepare_data(images)
        self.wrapper = SklearnWrapper(self.model)
        self.wrapper.fit(features, labels)
        return self
    
    def predict(self, images):
        """Predict SSC values."""
        if self.wrapper is None:
            raise RuntimeError("Model not fitted yet")
        features = self.prepare_data(images)
        return self.wrapper.predict(features)


class NDWIEmpirical:
    """
    NDWI-based empirical regression baseline.
    Uses Normalized Difference Water Index with empirical calibration.
    """
    def __init__(self, green_band=2, nir_band=7, calibration_factor=1.0):
        """
        Args:
            green_band: Index of green band (default 2 for Sentinel-2 B3)
            nir_band: Index of NIR band (default 7 for Sentinel-2 B8)
            calibration_factor: Empirical calibration factor
        """
        self.green_band = green_band
        self.nir_band = nir_band
        self.calibration_factor = calibration_factor
        self.a = 1.0  # Linear coefficient
        self.b = 0.0  # Bias
    
    def compute_ndwi(self, images):
        """
        Compute NDWI for each image.
        NDWI = (Green - NIR) / (Green + NIR)
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Extract bands
        green = images[:, self.green_band, :, :]
        nir = images[:, self.nir_band, :, :]
        
        # Compute NDWI
        ndwi = (green - nir) / (green + nir + 1e-8)
        
        # Average over spatial dimensions
        ndwi_mean = np.mean(ndwi, axis=(1, 2))
        return ndwi_mean
    
    def fit(self, images, labels):
        """
        Calibrate linear regression: SSC = a * NDWI + b
        """
        ndwi = self.compute_ndwi(images)
        
        # Simple linear regression
        n = len(ndwi)
        x_mean = np.mean(ndwi)
        y_mean = np.mean(labels)
        
        self.a = np.sum((ndwi - x_mean) * (labels - y_mean)) / np.sum((ndwi - x_mean) ** 2)
        self.b = y_mean - self.a * x_mean
        
        return self
    
    def predict(self, images):
        """Predict SSC from NDWI."""
        ndwi = self.compute_ndwi(images)
        ssc = self.a * ndwi + self.b
        return ssc * self.calibration_factor


class NIRRedRatio:
    """
    Simple NIR/Red ratio baseline.
    NIR/Red ratio is commonly used as a proxy for turbidity.
    """
    def __init__(self, red_band=3, nir_band=7, calibration_factor=1.0):
        """
        Args:
            red_band: Index of red band (default 3 for Sentinel-2 B4)
            nir_band: Index of NIR band (default 7 for Sentinel-2 B8)
            calibration_factor: Empirical calibration factor
        """
        self.red_band = red_band
        self.nir_band = nir_band
        self.calibration_factor = calibration_factor
        self.a = 1.0
        self.b = 0.0
    
    def compute_ratio(self, images):
        """
        Compute NIR/Red ratio for each image.
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        red = images[:, self.red_band, :, :]
        nir = images[:, self.nir_band, :, :]
        
        ratio = nir / (red + 1e-8)
        ratio_mean = np.mean(ratio, axis=(1, 2))
        return ratio_mean
    
    def fit(self, images, labels):
        """Calibrate linear regression."""
        ratio = self.compute_ratio(images)
        
        n = len(ratio)
        x_mean = np.mean(ratio)
        y_mean = np.mean(labels)
        
        self.a = np.sum((ratio - x_mean) * (labels - y_mean)) / np.sum((ratio - x_mean) ** 2)
        self.b = y_mean - self.a * x_mean
        
        return self
    
    def predict(self, images):
        """Predict SSC from NIR/Red ratio."""
        ratio = self.compute_ratio(images)
        ssc = self.a * ratio + self.b
        return ssc * self.calibration_factor


class MLModelWrapper(nn.Module):
    """
    PyTorch wrapper for sklearn/xgboost/lightgbm models.
    Allows using ML baselines in PyTorch training pipeline.
    """
    def __init__(self, ml_model):
        super().__init__()
        self.ml_model = ml_model
        self.device = 'cpu'
    
    def forward(self, x):
        """Forward pass - delegates to ML model."""
        predictions = self.ml_model.predict(x)
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float32)
        return predictions.view(-1, 1)
    
    def fit(self, train_loader):
        """
        Fit using a DataLoader.
        """
        all_images = []
        all_labels = []
        
        for images, labels in train_loader:
            all_images.append(images)
            all_labels.append(labels)
        
        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()
        
        self.ml_model.fit(images, labels)
        return self
