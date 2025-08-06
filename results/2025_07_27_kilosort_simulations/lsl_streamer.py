import numpy as np
import struct
import json
from pathlib import Path
import time
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import queue

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
        # For IMEC probes, AP channels are typically 0 to 383
        # This might vary based on your specific probe configuration
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
        # Get probe type specific gain
        imro_table = self.meta.get('imroTbl', '')
        
        # Default values for different probe types
        if 'imec' in self.bin_path.name.lower():
            # IMEC probe typical values
            max_int = 512
            v_range = 0.6  # 0.6V range typical for IMEC
        else:
            max_int = 32768
            v_range = 10.0
            
        return v_range / max_int * 1e6  # Convert to microvolts
    
    def get_data_chunk(self, start_sample, n_samples, channels=None):
        """Get a chunk of data"""
        if channels is None:
            channels = self.ap_channels  # Default to AP channels
        
        end_sample = min(start_sample + n_samples, self.n_samples)
        data_chunk = self.data[start_sample:end_sample, channels]
        
        # Convert to microvolts
        return data_chunk * self.uv_per_bit


class LSLStreamer:
    """Stream SpikeGLX data via LSL"""
    
    def __init__(self, reader, chunk_size=1000, stream_name="SpikeGLX_Stream"):
        self.reader = reader
        self.chunk_size = chunk_size
        self.current_sample = 0
        self.is_streaming = False
        
        # Select channels to stream (e.g., first 32 AP channels)
        self.stream_channels = reader.ap_channels[:32]
        self.n_stream_channels = len(self.stream_channels)
        
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
        sleep_time = samples_per_push / self.reader.sample_rate
        
        while self.is_streaming and self.current_sample < self.reader.n_samples:
            # Get data chunk
            data = self.reader.get_data_chunk(
                self.current_sample, 
                samples_per_push,
                self.stream_channels
            )
            
            # Push to LSL
            for sample in data:
                self.outlet.push_sample(sample.astype(np.float32))
            
            # Put data in queue for visualization (don't block)
            try:
                self.data_queue.put_nowait(data)
            except queue.Full:
                pass
            
            # Update position
            self.current_sample += len(data)
            
            # Sleep to maintain real-time streaming
            time.sleep(sleep_time)
        
        print(f"Streaming finished. Total samples streamed: {self.current_sample}")


class StreamVisualizer:
    """Visualize the LSL stream"""
    
    def __init__(self, streamer, display_seconds=5, update_interval=100):
        self.streamer = streamer
        self.display_seconds = display_seconds
        self.update_interval = update_interval
        
        # Calculate buffer size
        self.buffer_size = int(display_seconds * streamer.reader.sample_rate)
        
        # Initialize buffers for each channel
        self.buffers = [deque(maxlen=self.buffer_size) 
                       for _ in range(streamer.n_stream_channels)]
        
        # Setup plot
        self.fig, self.axes = plt.subplots(
            min(8, streamer.n_stream_channels), 1, 
            figsize=(12, 8), 
            sharex=True
        )
        
        if streamer.n_stream_channels == 1:
            self.axes = [self.axes]
        
        # Initialize lines
        self.lines = []
        time_array = np.linspace(0, display_seconds, self.buffer_size)
        
        for i, ax in enumerate(self.axes):
            line, = ax.plot(time_array, np.zeros(self.buffer_size))
            self.lines.append(line)
            ax.set_ylabel(f'Ch {self.streamer.stream_channels[i]}')
            ax.set_ylim(-200, 200)  # microvolts
            ax.grid(True, alpha=0.3)
        
        self.axes[-1].set_xlabel('Time (s)')
        self.fig.suptitle('SpikeGLX LSL Stream')
        
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
        
        return self.lines
    
    def start(self):
        """Start visualization"""
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            interval=self.update_interval,
            blit=True, cache_frame_data=False
        )
        plt.show()


def main():
    """Main function to run the LSL streamer"""
    
    # Path to your .cbin file
    cbin_path = "/home/riwata/Documents/projects/leadboss/data/44729857_unzipped/sim.imec0.ap.cbin"
    
    try:
        # Initialize reader
        print(f"Loading {cbin_path}...")
        reader = SpikeGLXReader(cbin_path)
        
        print(f"File info:")
        print(f"  Channels: {reader.n_channels}")
        print(f"  Sample rate: {reader.sample_rate} Hz")
        print(f"  Duration: {reader.n_samples / reader.sample_rate:.2f} seconds")
        print(f"  AP channels: {len(reader.ap_channels)}")
        
        # Create streamer
        streamer = LSLStreamer(reader, chunk_size=1000)
        print(f"\nStreaming {streamer.n_stream_channels} channels via LSL...")
        
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