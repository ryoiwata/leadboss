import numpy as np
from scipy import signal
from collections import deque
import struct
import json
from pathlib import Path
import time
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue

class FixedPoint:
    """Fixed-point arithmetic implementation for low-power processing"""
    
    @staticmethod
    def to_fixed(value, integer_bits, fractional_bits):
        """Convert floating point to fixed point representation"""
        scale = 1 << fractional_bits
        return int(value * scale)
    
    @staticmethod
    def from_fixed(value, integer_bits, fractional_bits):
        """Convert fixed point to floating point"""
        scale = 1 << fractional_bits
        return value / scale
    
    @staticmethod
    def multiply_fixed(a, b, result_format):
        """Multiply two fixed point numbers"""
        # Simplified multiplication - in real implementation would handle overflow
        return a * b >> result_format['shift']

class PreprocessingPipeline:
    """Real-time preprocessing pipeline based on patent specifications"""
    
    def __init__(self, sample_rate=30000, n_channels=32, use_fixed_point=False):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.use_fixed_point = use_fixed_point
        
        # 1. Amplification parameters
        self.amplification_gain = 1.0  # Adjust based on your hardware
        
        # 2. Bandpass filter design (500Hz - 5kHz)
        self.bp_low = 500
        self.bp_high = 5000
        self.filter_order = 2  # 2nd order Butterworth as per patent
        
        # Design IIR Butterworth filter
        self.sos = signal.butter(self.filter_order, 
                                [self.bp_low, self.bp_high], 
                                btype='band', 
                                fs=sample_rate, 
                                output='sos')
        
        # Initialize filter states for each channel
        self.zi = np.zeros((n_channels, self.sos.shape[0], 2))
        
        # 3. MAD calculation parameters
        self.alpha = 0.001  # Update rate for MAD (α in patent)
        self.beta = 4.0     # Threshold multiplier (β in patent)
        
        # MAD buffers for each channel
        self.mad_estimates = np.zeros(n_channels)
        self.thresholds = np.zeros(n_channels)
        
        # Initialize MAD with reasonable defaults
        self.mad_estimates[:] = 10.0  # microvolts
        self.thresholds = self.mad_estimates * self.beta
        
        # Ring buffers for MAD calculation (using deque for efficiency)
        self.mad_buffer_size = int(0.1 * sample_rate)  # 100ms window
        self.mad_buffers = [deque(maxlen=self.mad_buffer_size) for _ in range(n_channels)]
        
        # Fixed-point formats (Q7.15 for MAD as per patent)
        if use_fixed_point:
            self.mad_fixed = np.zeros(n_channels, dtype=np.int32)
            self.threshold_fixed = np.zeros(n_channels, dtype=np.int32)
    
    def amplify(self, data):
        """Apply amplification to the signal"""
        return data * self.amplification_gain
    
    def bandpass_filter_channel(self, data, channel_idx):
        """Apply bandpass filter to a single channel with state preservation"""
        filtered, self.zi[channel_idx] = signal.sosfilt(
            self.sos, data, zi=self.zi[channel_idx]
        )
        return filtered
    
    def update_mad(self, filtered_data, channel_idx):
        """Update MAD estimate for adaptive thresholding"""
        # Add new samples to buffer
        self.mad_buffers[channel_idx].extend(filtered_data)
        
        if len(self.mad_buffers[channel_idx]) >= self.mad_buffer_size:
            # Calculate median
            buffer_array = np.array(self.mad_buffers[channel_idx])
            median = np.median(buffer_array)
            
            # Calculate MAD
            mad = np.median(np.abs(buffer_array - median))
            
            # Update MAD estimate using exponential smoothing
            # m̂ = m + α(|x| - m) from patent
            self.mad_estimates[channel_idx] = (
                self.mad_estimates[channel_idx] + 
                self.alpha * (mad - self.mad_estimates[channel_idx])
            )
            
            # Update threshold: thres = βm
            self.thresholds[channel_idx] = self.beta * self.mad_estimates[channel_idx]
            
            # Fixed-point conversion if enabled
            if self.use_fixed_point:
                # Q7.15 format as per patent
                self.mad_fixed[channel_idx] = FixedPoint.to_fixed(
                    self.mad_estimates[channel_idx], 7, 15
                )
                self.threshold_fixed[channel_idx] = FixedPoint.to_fixed(
                    self.thresholds[channel_idx], 7, 15
                )
    
    def process_chunk(self, data_chunk):
        """Process a chunk of multi-channel data"""
        # Ensure data is 2D (samples x channels)
        if data_chunk.ndim == 1:
            data_chunk = data_chunk.reshape(-1, 1)
        
        n_samples, n_channels = data_chunk.shape
        
        # Preallocate output
        filtered_data = np.zeros_like(data_chunk)
        
        # Process each channel
        for ch_idx in range(n_channels):
            # 1. Amplification
            amplified = self.amplify(data_chunk[:, ch_idx])
            
            # 2. Bandpass filtering
            filtered = self.bandpass_filter_channel(amplified, ch_idx)
            filtered_data[:, ch_idx] = filtered
            
            # 3. Update MAD for this channel
            self.update_mad(filtered, ch_idx)
        
        return filtered_data
    
    def get_thresholds(self):
        """Get current thresholds for all channels"""
        if self.use_fixed_point:
            # Convert from fixed-point
            return np.array([
                FixedPoint.from_fixed(t, 7, 15) 
                for t in self.threshold_fixed
            ])
        return self.thresholds.copy()
    
    def get_mad_estimates(self):
        """Get current MAD estimates for all channels"""
        if self.use_fixed_point:
            # Convert from fixed-point
            return np.array([
                FixedPoint.from_fixed(m, 7, 15) 
                for m in self.mad_fixed
            ])
        return self.mad_estimates.copy()

# Modified SpikeGLXReader class remains the same
class SpikeGLXReader:
    """Read SpikeGLX binary files"""
    
    def __init__(self, bin_path):
        self.bin_path = Path(bin_path)
        self.meta_path = self.bin_path.with_suffix('.meta')
        
        # Read metadata
        self.meta = self._read_meta()
        
        # Extract key parameters
        self.n_channels = int(self.meta['nSavedChans'])
        self.sample_rate = self._get_sample_rate()
        self.file_size_bytes = self.bin_path.stat().st_size
        self.n_samples = self.file_size_bytes // (2 * self.n_channels)  # int16 = 2 bytes
        
        # Channel info
        self.ap_channels = self._get_ap_channels()
        self.lfp_channels = self._get_lfp_channels()
        
        # Conversion factors
        self.uv_per_bit = self._get_uv_per_bit()
        
        # Memory map the file
        self.data = np.memmap(self.bin_path, dtype='int16', mode='r',
                             shape=(self.n_samples, self.n_channels))
    
    def _read_meta(self):
        """Read the metadata file"""
        meta = {}
        with open(self.meta_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = line.strip().split('=')
                    meta[key] = val
        return meta
    
    def _get_sample_rate(self):
        """Extract sample rate from metadata"""
        if 'imSampRate' in self.meta:
            return float(self.meta['imSampRate'])
        elif 'niSampRate' in self.meta:
            return float(self.meta['niSampRate'])
        else:
            raise ValueError("Cannot find sample rate in metadata")
    
    def _get_ap_channels(self):
        """Get AP band channel indices"""
        saved_chans = self.meta.get('snsApLfSy', '384,0,1')
        ap_count = int(saved_chans.split(',')[0])
        return list(range(ap_count))
    
    def _get_lfp_channels(self):
        """Get LFP band channel indices"""
        saved_chans = self.meta.get('snsApLfSy', '384,0,1')
        parts = saved_chans.split(',')
        if len(parts) > 1:
            lfp_count = int(parts[1])
            ap_count = int(parts[0])
            return list(range(ap_count, ap_count + lfp_count))
        return []
    
    def _get_uv_per_bit(self):
        """Calculate microvolts per bit for int16 to voltage conversion"""
        if 'imec' in self.bin_path.name.lower():
            max_int = 512
            v_range = 0.6  # 0.6V range typical for IMEC
        else:
            max_int = 32768
            v_range = 10.0
            
        return v_range / max_int * 1e6  # Convert to microvolts
    
    def get_data_chunk(self, start_sample, n_samples, channels=None):
        """Get a chunk of data"""
        if channels is None:
            channels = self.ap_channels
        
        end_sample = min(start_sample + n_samples, self.n_samples)
        
        # Ensure channels is a list for proper indexing
        if isinstance(channels, list):
            data_chunk = self.data[start_sample:end_sample][:, channels]
        else:
            data_chunk = self.data[start_sample:end_sample, channels]
        
        # Convert to microvolts
        return data_chunk * self.uv_per_bit

# Modified LSLStreamer with preprocessing
class LSLStreamer:
    """Stream SpikeGLX data via LSL with preprocessing"""
    
    def __init__(self, reader, chunk_size=1000, stream_name="SpikeGLX_Stream", 
                 use_preprocessing=True, use_fixed_point=False, playback_speed=1.0):
        self.reader = reader
        self.chunk_size = chunk_size
        self.current_sample = 0
        self.is_streaming = False
        self.playback_speed = playback_speed
        self.use_preprocessing = use_preprocessing
        
        # Select channels to stream
        self.stream_channels = reader.ap_channels[:32]
        self.n_stream_channels = len(self.stream_channels)
        
        # Initialize preprocessing pipeline
        if use_preprocessing:
            self.preprocessor = PreprocessingPipeline(
                sample_rate=reader.sample_rate,
                n_channels=self.n_stream_channels,
                use_fixed_point=use_fixed_point
            )
        
        # Create LSL stream info
        info = StreamInfo(
            name=stream_name,
            type='EEG',
            channel_count=self.n_stream_channels,
            nominal_srate=reader.sample_rate,
            channel_format='float32',
            source_id=str(reader.bin_path)
        )
        
        # Add channel labels
        channels = info.desc().append_child("channels")
        for i, ch_idx in enumerate(self.stream_channels):
            ch = channels.append_child("channel")
            ch.append_child_value("label", f"AP_{ch_idx}")
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        
        # Create outlet
        self.outlet = StreamOutlet(info)
        
        # Queue for visualization
        self.data_queue = queue.Queue(maxsize=10)
        
        # Queue for threshold monitoring
        self.threshold_queue = queue.Queue(maxsize=10)
    
    def start_streaming(self):
        """Start streaming data"""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.start()
    
    def stop_streaming(self):
        """Stop streaming data"""
        self.is_streaming = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join()
    
    def _stream_loop(self):
        """Main streaming loop"""
        samples_per_push = int(self.chunk_size)
        sleep_time = samples_per_push / self.reader.sample_rate / self.playback_speed
        
        while self.is_streaming and self.current_sample < self.reader.n_samples:
            # Get data chunk
            data = self.reader.get_data_chunk(
                self.current_sample, 
                samples_per_push,
                self.stream_channels
            )
            
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                data = self.preprocessor.process_chunk(data)
                
                # Periodically update threshold information
                if self.current_sample % (self.reader.sample_rate // 10) == 0:  # 10Hz update
                    try:
                        self.threshold_queue.put_nowait({
                            'thresholds': self.preprocessor.get_thresholds(),
                            'mad_estimates': self.preprocessor.get_mad_estimates()
                        })
                    except queue.Full:
                        pass
            
            # Push to LSL with adjusted timestamps if needed
            if self.playback_speed != 1.0:
                timestamp = time.time()
                for i in range(len(data)):
                    sample_offset = i / self.reader.sample_rate / self.playback_speed
                    self.outlet.push_sample(data[i].astype(np.float32), timestamp + sample_offset)
            else:
                for i in range(len(data)):
                    self.outlet.push_sample(data[i].astype(np.float32))
            
            # Put data in queue for visualization
            try:
                self.data_queue.put_nowait(data)
            except queue.Full:
                pass
            
            # Update position
            self.current_sample += len(data)
            
            # Sleep to maintain desired playback speed
            time.sleep(sleep_time)
        
        print(f"Streaming finished. Total samples streamed: {self.current_sample}")

# Enhanced visualizer with threshold display
# Enhanced visualizer with threshold display (fixed version)
class StreamVisualizer:
    """Visualize the LSL stream with threshold information"""
    
    def __init__(self, streamer, display_seconds=0.01, update_interval=100):
        self.streamer = streamer
        self.display_seconds = display_seconds
        self.update_interval = update_interval
        
        # Calculate buffer size
        self.buffer_size = int(display_seconds * streamer.reader.sample_rate)
        
        # Initialize buffers for each channel
        self.buffers = [deque(maxlen=self.buffer_size) 
                       for _ in range(streamer.n_stream_channels)]
        
        # Setup plot with additional threshold subplot
        n_display_channels = min(8, streamer.n_stream_channels)
        
        if streamer.use_preprocessing:
            self.fig = plt.figure(figsize=(14, 10))
            gs = self.fig.add_gridspec(n_display_channels + 1, 2, 
                                      width_ratios=[3, 1], 
                                      height_ratios=[1]*n_display_channels + [0.5])
            
            # Signal plots
            self.axes = []
            for i in range(n_display_channels):
                ax = self.fig.add_subplot(gs[i, 0])
                self.axes.append(ax)
            
            # Threshold plot
            self.threshold_ax = self.fig.add_subplot(gs[:, 1])
        else:
            self.fig, self.axes = plt.subplots(
                n_display_channels, 1, 
                figsize=(12, 8), 
                sharex=True
            )
            if n_display_channels == 1:
                self.axes = [self.axes]
        
        # Initialize lines
        self.lines = []
        self.threshold_lines = []
        self.neg_threshold_lines = []  # Initialize this list here!
        time_array = np.linspace(0, display_seconds, self.buffer_size)
        
        for i, ax in enumerate(self.axes):
            line, = ax.plot(time_array, np.zeros(self.buffer_size))
            self.lines.append(line)
            ax.set_ylabel(f'Ch {self.streamer.stream_channels[i]}')
            ax.set_ylim(-200, 200)  # microvolts
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines if preprocessing is enabled
            if self.streamer.use_preprocessing:
                # Positive threshold line
                thresh_line = ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.threshold_lines.append(thresh_line)
                
                # Negative threshold line
                neg_thresh_line = ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.neg_threshold_lines.append(neg_thresh_line)
        
        self.axes[-1].set_xlabel('Time (s)')
        self.fig.suptitle('SpikeGLX LSL Stream with Preprocessing')
        
        # Setup threshold bar plot
        if self.streamer.use_preprocessing:
            self.threshold_bars = None
            self.mad_bars = None
            self._setup_threshold_plot()
    
    def _setup_threshold_plot(self):
        """Setup the threshold visualization plot"""
        self.threshold_ax.clear()
        channels = range(min(8, self.streamer.n_stream_channels))
        x = np.arange(len(channels))
        
        self.threshold_bars = self.threshold_ax.bar(x - 0.2, np.zeros(len(channels)), 
                                                   0.4, label='Threshold', alpha=0.7)
        self.mad_bars = self.threshold_ax.bar(x + 0.2, np.zeros(len(channels)), 
                                             0.4, label='MAD', alpha=0.7)
        
        self.threshold_ax.set_xlabel('Channel')
        self.threshold_ax.set_ylabel('µV')
        self.threshold_ax.set_title('Adaptive Thresholds')
        self.threshold_ax.legend()
        self.threshold_ax.grid(True, alpha=0.3)
    
    def update_plot(self, frame):
        """Update plot with new data"""
        # Get all available data from queue
        new_data = []
        while not self.streamer.data_queue.empty():
            try:
                data = self.streamer.data_queue.get_nowait()
                new_data.append(data)
            except queue.Empty:
                break
        
        if new_data:
            # Concatenate all new data
            combined_data = np.vstack(new_data)
            
            # Update buffers
            for ch_idx in range(self.streamer.n_stream_channels):
                self.buffers[ch_idx].extend(combined_data[:, ch_idx])
            
            # Update plots
            for i in range(min(len(self.axes), self.streamer.n_stream_channels)):
                if len(self.buffers[i]) > 0:
                    y_data = np.array(self.buffers[i])
                    # Pad with zeros if buffer not full
                    if len(y_data) < self.buffer_size:
                        y_data = np.pad(y_data, (self.buffer_size - len(y_data), 0))
                    
                    self.lines[i].set_ydata(y_data)
                    
                    # Auto-scale y-axis
                    if len(y_data) > 0:
                        y_min, y_max = np.min(y_data), np.max(y_data)
                        margin = (y_max - y_min) * 0.1
                        self.axes[i].set_ylim(y_min - margin, y_max + margin)
        
        # Update threshold visualization if preprocessing is enabled
        if self.streamer.use_preprocessing:
            try:
                threshold_info = self.streamer.threshold_queue.get_nowait()
                thresholds = threshold_info['thresholds']
                mad_estimates = threshold_info['mad_estimates']
                
                # Update threshold lines on signal plots
                for i in range(min(len(thresholds), len(self.threshold_lines))):
                    thresh = thresholds[i]
                    
                    # Update positive threshold line
                    self.threshold_lines[i].set_ydata([thresh, thresh])
                    
                    # Update negative threshold line
                    self.neg_threshold_lines[i].set_ydata([-thresh, -thresh])
                
                # Update bar plots
                n_channels = min(8, len(thresholds))
                for i in range(n_channels):
                    self.threshold_bars[i].set_height(thresholds[i])
                    self.mad_bars[i].set_height(mad_estimates[i])
                
                self.threshold_ax.relim()
                self.threshold_ax.autoscale_view()
                
            except queue.Empty:
                pass
        
        return self.lines
    
    def start(self):
        """Start visualization"""
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            interval=self.update_interval,
            blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the LSL streamer with preprocessing"""
    
    # Path to your .cbin file
    cbin_path = "/home/riwata/Documents/projects/leadboss/data/44729857_unzipped/sim.imec0.ap.cbin"
    
    # Configuration options
    use_preprocessing = True  # Enable/disable preprocessing
    use_fixed_point = False   # Enable/disable fixed-point arithmetic
    playback_speed = 0.01      # Playback speed multiplier
    
    try:
        # Initialize reader
        print(f"Loading {cbin_path}...")
        reader = SpikeGLXReader(cbin_path)
        
        print(f"File info:")
        print(f"  Channels: {reader.n_channels}")
        print(f"  Sample rate: {reader.sample_rate} Hz")
        print(f"  Duration: {reader.n_samples / reader.sample_rate:.2f} seconds")
        print(f"  AP channels: {len(reader.ap_channels)}")
        
        # Create streamer with preprocessing
        streamer = LSLStreamer(
            reader, 
            chunk_size=1000, 
            use_preprocessing=use_preprocessing,
            use_fixed_point=use_fixed_point,
            playback_speed=playback_speed
        )
        
        print(f"\nStreaming {streamer.n_stream_channels} channels via LSL...")
        print(f"Preprocessing: {'Enabled' if use_preprocessing else 'Disabled'}")
        print(f"Fixed-point arithmetic: {'Enabled' if use_fixed_point else 'Disabled'}")
        print(f"Playback speed: {playback_speed}x")
        
        if use_preprocessing:
            print("\nPreprocessing pipeline:")
            print(f"  - Amplification: {streamer.preprocessor.amplification_gain}x")
            print(f"  - Bandpass filter: {streamer.preprocessor.bp_low}-{streamer.preprocessor.bp_high} Hz")
            print(f"  - MAD threshold: {streamer.preprocessor.beta}x MAD")
        
        # Start streaming
        streamer.start_streaming()
        
        # Create and start visualizer
        visualizer = StreamVisualizer(streamer)
        
        print("\nStreaming started. Close the plot window to stop.")
        visualizer.start()
        
        # Stop streaming when plot is closed
        streamer.stop_streaming()
        
    except FileNotFoundError:
        print(f"Error: Could not find {cbin_path}")
        print("Make sure the .cbin and .meta files are in the current directory")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()