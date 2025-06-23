import numpy as np
from scipy import signal
from collections import deque
import threading
import queue
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings

@dataclass
class SpikeDetection:
    """Data class for spike detection results"""
    channel: int
    timestamp: float
    spike_category: int
    lv: float
    mv: float
    rv: float
    lp: float
    mp: float
    rp: float
    
@dataclass
class BOSSConfig:
    """Configuration parameters for BOSS algorithm"""
    # Sampling parameters
    sampling_rate: float = 20000  # Hz
    
    # Bandpass filter parameters
    bp_low: float = 500  # Hz
    bp_high: float = 5000  # Hz
    filter_order: int = 2
    
    # Brown's exponential smoothing
    alpha: float = 0.1  # Smoothing factor
    
    # MAD parameters
    mad_alpha: float = 0.0002  # Update multiplier
    mad_scale: float = 4.0  # Threshold scaling factor
    
    # Spike detection thresholds
    max_ldist: float = 0.002  # seconds
    min_rdist: float = 0.0005  # seconds
    min_ratio: float = 0.5
    max_asym: float = 2.0
    max_cost: float = 0.3
    
    # Buffer sizes
    signal_buffer_size: int = 100  # samples
    mad_buffer_size: int = 1000  # samples for MAD calculation

class BrownDoubleExponentialSmoothing:
    """Brown's double exponential smoothing implementation"""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.s1 = None
        self.s2 = None
        
    def update(self, x: float) -> float:
        if self.s1 is None:
            self.s1 = x
            self.s2 = x
            return x
            
        self.s1 = self.alpha * x + (1 - self.alpha) * self.s1
        self.s2 = self.alpha * self.s1 + (1 - self.alpha) * self.s2
        
        # Return the smoothed value
        return 2 * self.s1 - self.s2
    
    def reset(self):
        self.s1 = None
        self.s2 = None

class ChannelProcessor:
    """Process a single channel for spike detection"""
    def __init__(self, channel_id: int, config: BOSSConfig):
        self.channel_id = channel_id
        self.config = config
        
        # Butterworth bandpass filter
        nyquist = config.sampling_rate / 2
        self.sos = signal.butter(
            config.filter_order,
            [config.bp_low / nyquist, config.bp_high / nyquist],
            btype='band',
            output='sos'
        )
        self.zi = signal.sosfilt_zi(self.sos)
        
        # Smoothing
        self.smoother = BrownDoubleExponentialSmoothing(config.alpha)
        
        # Buffers
        self.signal_buffer = deque(maxlen=config.signal_buffer_size)
        self.smoothed_buffer = deque(maxlen=config.signal_buffer_size)
        self.mad_buffer = deque(maxlen=config.mad_buffer_size)
        
        # MAD estimation
        self.mad_estimate = 0.0
        self.threshold = 0.0
        
        # State for peak/valley detection
        self.last_peak_idx = -1
        self.last_valley_idx = -1
        self.potential_spike = None
        
    def process_sample(self, sample: float, timestamp: float) -> Optional[SpikeDetection]:
        """Process a single sample and return spike if detected"""
        # Apply bandpass filter
        filtered, self.zi = signal.sosfilt(self.sos, [sample], zi=self.zi)
        filtered_value = filtered[0]
        
        # Apply smoothing
        smoothed_value = self.smoother.update(filtered_value)
        
        # Update buffers
        self.signal_buffer.append(filtered_value)
        self.smoothed_buffer.append(smoothed_value)
        
        # Update MAD estimate
        self._update_mad(filtered_value)
        
        # Check for spike
        if len(self.signal_buffer) >= 20:  # Need minimum samples
            spike = self._detect_spike(timestamp)
            if spike:
                return spike
                
        return None
    
    def _update_mad(self, value: float):
        """Update Mean Absolute Deviation estimate"""
        self.mad_buffer.append(value)
        
        if len(self.mad_buffer) > 10:
            # Compute median
            median = np.median(list(self.mad_buffer))
            
            # Update MAD estimate
            abs_dev = abs(value - median)
            self.mad_estimate = (self.mad_estimate + 
                               self.config.mad_alpha * (abs_dev - self.mad_estimate))
            
            # Update threshold
            self.threshold = self.config.mad_scale * self.mad_estimate
    
    def _find_peaks_valleys(self) -> Tuple[List[int], List[int]]:
        """Find local peaks and valleys in the smoothed buffer"""
        if len(self.smoothed_buffer) < 3:
            return [], []
            
        data = np.array(self.smoothed_buffer)
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        
        # Find valleys (local minima)
        valleys = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                valleys.append(i)
                
        return peaks, valleys
    
    def _detect_spike(self, timestamp: float) -> Optional[SpikeDetection]:
        """Detect spike based on characteristic pattern"""
        peaks, valleys = self._find_peaks_valleys()
        
        if not peaks or not valleys:
            return None
            
        # Look for the characteristic spike pattern:
        # peak -> valley -> peak with valley being the deepest
        
        for valley_idx in valleys:
            # Find left peak (before valley)
            left_peaks = [p for p in peaks if p < valley_idx]
            if not left_peaks:
                continue
            left_peak_idx = left_peaks[-1]  # Closest peak before valley
            
            # Find right peak (after valley)
            right_peaks = [p for p in peaks if p > valley_idx]
            if not right_peaks:
                continue
            right_peak_idx = right_peaks[0]  # Closest peak after valley
            
            # Extract fit values
            lv = self.signal_buffer[left_peak_idx]
            mv = self.signal_buffer[valley_idx]
            rv = self.signal_buffer[right_peak_idx]
            
            # Convert indices to time
            dt = 1.0 / self.config.sampling_rate
            lp = (left_peak_idx - len(self.signal_buffer) + 1) * dt
            mp = (valley_idx - len(self.signal_buffer) + 1) * dt
            rp = (right_peak_idx - len(self.signal_buffer) + 1) * dt
            
            # Compute characteristic values
            ldist = mp - lp
            rdist = rp - mp
            
            if abs(mv) > 0:
                ratio = rv / abs(mv)
                cost = self._compute_cost(left_peak_idx, right_peak_idx) / abs(mv)
            else:
                continue
                
            if rv > 0:
                asym = lv / rv
            else:
                continue
            
            # Check thresholds
            if (abs(mv) > self.threshold and  # Significant deflection
                ldist < self.config.max_ldist and
                rdist > self.config.min_rdist and
                ratio > self.config.min_ratio and
                asym < self.config.max_asym and
                cost < self.config.max_cost):
                
                # Classify spike (simple binning based on rdist)
                spike_category = self._classify_spike(rdist)
                
                return SpikeDetection(
                    channel=self.channel_id,
                    timestamp=timestamp,
                    spike_category=spike_category,
                    lv=lv, mv=mv, rv=rv,
                    lp=lp, mp=mp, rp=rp
                )
                
        return None
    
    def _compute_cost(self, start_idx: int, end_idx: int) -> float:
        """Compute noise proxy (cost) as accumulated error"""
        if end_idx >= len(self.signal_buffer) or start_idx < 0:
            return float('inf')
            
        signal_segment = list(self.signal_buffer)[start_idx:end_idx+1]
        smoothed_segment = list(self.smoothed_buffer)[start_idx:end_idx+1]
        
        if len(signal_segment) != len(smoothed_segment):
            return float('inf')
            
        # Accumulated absolute deviation
        error = sum(abs(s - sm) for s, sm in zip(signal_segment, smoothed_segment))
        return error
    
    def _classify_spike(self, rdist: float) -> int:
        """Classify spike into categories based on rdist"""
        # Simple binning into 6 categories
        bins = np.linspace(self.config.min_rdist, self.config.max_ldist, 7)
        category = np.digitize(rdist, bins)
        return min(category, 6)  # Cap at 6 categories

class BOSSMultiChannelProcessor:
    """Multi-channel BOSS processor for real-time spike detection"""
    def __init__(self, num_channels: int, config: BOSSConfig = None):
        self.num_channels = num_channels
        self.config = config or BOSSConfig()
        
        # Create channel processors
        self.channels = [
            ChannelProcessor(i, self.config) 
            for i in range(num_channels)
        ]
        
        # Output queue for detected spikes
        self.spike_queue = queue.Queue()
        
        # Processing statistics
        self.samples_processed = 0
        self.spikes_detected = 0
        
    def process_frame(self, data: np.ndarray, timestamp: float) -> List[SpikeDetection]:
        """Process a frame of multi-channel data
        
        Args:
            data: Array of shape (num_channels,) containing one sample per channel
            timestamp: Timestamp for this frame
            
        Returns:
            List of detected spikes
        """
        if data.shape[0] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {data.shape[0]}")
            
        spikes = []
        
        for channel_idx, sample in enumerate(data):
            spike = self.channels[channel_idx].process_sample(sample, timestamp)
            if spike:
                spikes.append(spike)
                self.spikes_detected += 1
                
        self.samples_processed += 1
        
        # Put spikes in queue for asynchronous processing
        for spike in spikes:
            self.spike_queue.put(spike)
            
        return spikes
    
    def stream_process(self, data_generator):
        """Process streaming data from a generator
        
        Args:
            data_generator: Generator yielding (data, timestamp) tuples
        """
        for data, timestamp in data_generator:
            self.process_frame(data, timestamp)
            
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            'samples_processed': self.samples_processed,
            'spikes_detected': self.spikes_detected,
            'detection_rate': self.spikes_detected / max(1, self.samples_processed)
        }

# Example usage and testing
def example_data_generator(num_channels: int, duration: float, sampling_rate: float):
    """Generate example data with synthetic spikes"""
    num_samples = int(duration * sampling_rate)
    dt = 1.0 / sampling_rate
    
    for i in range(num_samples):
        timestamp = i * dt
        
        # Generate base noise
        data = np.random.randn(num_channels) * 0.05
        
        # Add synthetic spike occasionally
        if np.random.rand() < 0.01:  # 1% chance per sample
            channel = np.random.randint(num_channels)
            
            # Create spike-like waveform
            spike_amp = np.random.uniform(0.5, 1.5)
            data[channel] += spike_amp * np.sin(2 * np.pi * 3000 * timestamp)
            
        yield data, timestamp

# Real-time processing example
class RealTimeProcessor(threading.Thread):
    """Thread for real-time spike processing and output"""
    def __init__(self, boss_processor: BOSSMultiChannelProcessor):
        super().__init__()
        self.boss = boss_processor
        self.running = False
        
    def run(self):
        """Process spikes from queue"""
        self.running = True
        while self.running:
            try:
                spike = self.boss.spike_queue.get(timeout=0.1)
                print(f"Spike detected - Channel: {spike.channel}, "
                      f"Time: {spike.timestamp:.4f}, Category: {spike.spike_category}")
            except queue.Empty:
                continue
                
    def stop(self):
        self.running = False

if __name__ == "__main__":
    # Example usage
    num_channels = 256
    sampling_rate = 20000  # Hz
    
    # Create BOSS processor
    config = BOSSConfig(sampling_rate=sampling_rate)
    boss = BOSSMultiChannelProcessor(num_channels, config)
    
    # Create real-time output thread
    output_thread = RealTimeProcessor(boss)
    output_thread.start()
    
    # Process example data
    print("Starting real-time spike detection...")
    try:
        for data, timestamp in example_data_generator(num_channels, 10.0, sampling_rate):
            boss.process_frame(data, timestamp)
            
            # Print statistics periodically
            if boss.samples_processed % 10000 == 0:
                stats = boss.get_statistics()
                print(f"Processed {stats['samples_processed']} samples, "
                      f"detected {stats['spikes_detected']} spikes")
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
    output_thread.stop()
    output_thread.join()
    
    # Final statistics
    stats = boss.get_statistics()
    print(f"\nFinal statistics:")
    print(f"Total samples: {stats['samples_processed']}")
    print(f"Total spikes: {stats['spikes_detected']}")
    print(f"Detection rate: {stats['detection_rate']:.4f}")