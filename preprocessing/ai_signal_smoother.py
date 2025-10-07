import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Optional
from functools import lru_cache

class AISignalSmoother:
    """AI-powered signal smoother for professional EEG visualization"""
    
    def __init__(self, sampling_rate: float = 256):
        self.sampling_rate = sampling_rate
        self._filter_cache = {}
    
    def smooth_signals(self, data: np.ndarray, method: str = 'savitzky_golay') -> np.ndarray:
        """
        Apply intelligent smoothing to EEG signals
        
        Args:
            data: Input signals (channels x samples)
            method: Smoothing method ('savitzky_golay', 'gaussian', 'moving_average', 'adaptive')
        
        Returns:
            Smoothed signals
        """
        if method == 'savitzky_golay':
            return self._savitzky_golay_smooth(data)
        elif method == 'gaussian':
            return self._gaussian_smooth(data)
        elif method == 'moving_average':
            return self._moving_average_smooth(data)
        elif method == 'adaptive':
            return self._adaptive_smooth(data)
        else:
            return self._savitzky_golay_smooth(data)
    
    def _savitzky_golay_smooth(self, data: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
        """
        Optimized Savitzky-Golay filter with vectorization
        This preserves peaks and important features while smoothing noise
        """
        if window_length % 2 == 0:
            window_length += 1
        
        window_length = min(window_length, data.shape[1] - 1)
        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        smoothed = np.apply_along_axis(
            lambda x: signal.savgol_filter(x, window_length, polyorder), 
            axis=1, 
            arr=data
        )
        
        return smoothed
    
    def _gaussian_smooth(self, data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Optimized Gaussian smoothing using vectorized operations
        """
        smoothed = np.apply_along_axis(
            lambda x: gaussian_filter1d(x, sigma=sigma), 
            axis=1, 
            arr=data
        )
        
        return smoothed
    
    def _moving_average_smooth(self, data: np.ndarray, window_size: int = 15) -> np.ndarray:
        """
        Apply moving average for simple smoothing
        """
        smoothed = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed[i] = np.convolve(data[i], np.ones(window_size)/window_size, mode='same')
        
        return smoothed
    
    def _adaptive_smooth(self, data: np.ndarray) -> np.ndarray:
        """
        AI-adaptive smoothing based on signal characteristics
        Uses different smoothing strength based on signal variance
        """
        smoothed = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            local_variance = self._compute_local_variance(channel_data)
            
            high_var_regions = local_variance > np.median(local_variance)
            low_var_regions = ~high_var_regions
            
            temp_smooth = np.copy(channel_data)
            temp_smooth[high_var_regions] = signal.savgol_filter(
                channel_data, min(51, len(channel_data)//4 if len(channel_data) > 100 else 11), 3
            )[high_var_regions]
            temp_smooth[low_var_regions] = gaussian_filter1d(
                channel_data, sigma=1.5
            )[low_var_regions]
            
            smoothed[i] = temp_smooth
        
        return smoothed
    
    def _compute_local_variance(self, data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Compute local variance for adaptive smoothing"""
        variance = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window)
            variance[i] = np.var(data[start:end])
        
        return variance
    
    def enhance_visualization(self, data: np.ndarray, 
                            baseline_correct: bool = True,
                            smooth_level: str = 'medium') -> np.ndarray:
        """
        Complete AI enhancement for professional visualization
        
        Args:
            data: Input signals
            baseline_correct: Remove baseline drift
            smooth_level: 'light', 'medium', or 'heavy'
        
        Returns:
            Enhanced signals ready for display
        """
        enhanced = np.copy(data)
        
        if baseline_correct:
            enhanced = self._remove_baseline_drift(enhanced)
        
        if smooth_level == 'light':
            enhanced = self._savitzky_golay_smooth(enhanced, window_length=11, polyorder=2)
        elif smooth_level == 'medium':
            enhanced = self._savitzky_golay_smooth(enhanced, window_length=21, polyorder=3)
        elif smooth_level == 'heavy':
            enhanced = self._adaptive_smooth(enhanced)
        
        enhanced = self._suppress_outliers(enhanced)
        
        return enhanced
    
    def _remove_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """Remove slow baseline drift for cleaner visualization"""
        corrected = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            b, a = signal.butter(3, 0.5 / (self.sampling_rate / 2), btype='high')
            corrected[i] = signal.filtfilt(b, a, data[i])
        
        return corrected
    
    def _suppress_outliers(self, data: np.ndarray, threshold: float = 4.0) -> np.ndarray:
        """Suppress extreme outliers that can make signals look messy"""
        suppressed = np.copy(data)
        
        for i in range(data.shape[0]):
            channel = data[i]
            median = np.median(channel)
            mad = np.median(np.abs(channel - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (channel - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
                suppressed[i][outliers] = median
        
        return suppressed
    
    def real_time_smooth(self, data: np.ndarray, buffer: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ultra-fast smoothing for real-time streaming with enhanced signal quality
        Uses optimized Savitzky-Golay filter for minimal latency
        """
        cache_key = f"sg_{data.shape}"
        if cache_key not in self._filter_cache:
            self._filter_cache[cache_key] = (11, 2)
        
        return self._savitzky_golay_smooth(data, window_length=11, polyorder=2)
