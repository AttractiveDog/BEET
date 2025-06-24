#!/usr/bin/env python3
"""
BEET - Virtual Microphone Application
A Python application that creates a virtual microphone with GUI interface.
"""

import sys
import logging
import subprocess
import threading
import time
from typing import Optional, List, Dict
import webbrowser
from PyQt5.QtWidgets import QInputDialog

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                 QWidget, QPushButton, QLabel, QSlider, QTextEdit, 
                                 QListWidget, QGroupBox, QMessageBox, QStatusBar,
                                 QSplitter, QFrame, QComboBox, QProgressBar, QTabWidget, QCheckBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    print("PyQt5 not found. Please install it with: pip install PyQt5")
    sys.exit(1)

try:
    import pulsectl
except ImportError:
    print("pulsectl not found. Please install it with: pip install pulsectl")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not found. Please install it with: pip install numpy")
    sys.exit(1)

try:
    import pyttsx3
except ImportError:
    print("pyttsx3 not found. Please install it with: pip install pyttsx3")
    sys.exit(1)

import io
import tempfile
import os
import wave
import time
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Import TTS dependencies with proper error handling
GTTS_AVAILABLE = True
PYGAME_AVAILABLE = True
GROQ_AVAILABLE = True
SOUNDDEVICE_AVAILABLE = True


try:
    from gtts import gTTS
    print("✅ gTTS loaded successfully")
except ImportError as e:
    print(f"⚠️  gTTS not available: {e}")
    GTTS_AVAILABLE = False

try:
    import pygame
    print("✅ pygame loaded successfully")
except ImportError as e:
    print(f"⚠️  pygame not available: {e}")
    PYGAME_AVAILABLE = False

try:
    from groq import Groq
    print("✅ Groq API loaded successfully")
except ImportError as e:
    print(f"⚠️  Groq API not available: {e}")
    GROQ_AVAILABLE = False

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    print("✅ SoundDevice loaded successfully")
except ImportError as e:
    print(f"⚠️  SoundDevice not available: {e}")
    SOUNDDEVICE_AVAILABLE = False



# GLOBAL LOGGING CONFIG
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)





class GroqTextGenerationManager:
    """Manages text generation using Groq's Chat Completions API"""
    
    def __init__(self):
        self.groq_client = None
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant. Provide concise, relevant responses to user queries."
        self.max_history_length = 10  # Keep last 10 exchanges
        self.generation_callback = None
        self.init_groq_client()
    
    def init_groq_client(self):
        """Initialize Groq client for text generation"""
        try:
            if not GROQ_AVAILABLE:
                logging.error("Groq package not available for text generation")
                return False
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logging.error("GROQ_API_KEY not found in environment")
                return False
            
            # Debug: Show first and last few characters of API key
            api_key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else api_key
            logging.info(f"🔑 Using Groq API key: {api_key_preview}")
            
            self.groq_client = Groq(api_key=api_key)
            logging.info("✅ Groq Text Generation API loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing Groq text generation client: {e}")
            return False
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for text generation"""
        self.system_prompt = prompt
        logging.info(f"System prompt updated: {prompt[:50]}...")
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input using Groq"""
        try:
            if not self.groq_client:
                return "Error: Groq client not initialized"
            
            if not user_input.strip():
                return ""
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-self.max_history_length:])
            
            # Generate response using Groq
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
                stream=False
            )
            
            response = chat_completion.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Trim conversation history if too long
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            logging.info(f"💬 Generated response: {response[:50]}...")
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response_streaming(self, user_input: str, callback=None):
        """Generate a streaming response to user input using Groq"""
        try:
            if not self.groq_client:
                if callback:
                    callback("Error: Groq client not initialized")
                return
            
            if not user_input.strip():
                return
            
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Prepare messages for API call
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-self.max_history_length:])
            
            # Generate streaming response using Groq
            stream = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
                stream=True
            )
            
            response_chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    response_chunks.append(chunk_content)
                    if callback:
                        callback(chunk_content, is_complete=False)
            
            # Complete response
            full_response = ''.join(response_chunks)
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            # Trim conversation history if too long
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            if callback:
                callback("", is_complete=True)
            
            logging.info(f"💬 Generated streaming response: {full_response[:50]}...")
            
        except Exception as e:
            logging.error(f"Error generating streaming response: {e}")
            if callback:
                callback(f"Error: {str(e)}", is_complete=True)
    
    def generate_independent_response_streaming(self, user_input: str, callback=None):
        """Generate a streaming response to user input without conversation history"""
        try:
            if not self.groq_client:
                if callback:
                    callback("Error: Groq client not initialized")
                return
            
            if not user_input.strip():
                return
            
            # Prepare messages for API call - only system prompt and current user input
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Generate streaming response using Groq
            stream = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
                stream=True
            )
            
            response_chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    response_chunks.append(chunk_content)
                    if callback:
                        callback(chunk_content, is_complete=False)
            
            # Complete response
            full_response = ''.join(response_chunks)
            
            if callback:
                callback("", is_complete=True)
            
            logging.info(f"💬 Generated independent response: {full_response[:50]}...")
            
        except Exception as e:
            logging.error(f"Error generating independent response: {e}")
            if callback:
                callback(f"Error: {str(e)}", is_complete=True)
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logging.info("🗑️ Conversation history cleared")


class GroqSTTManager:
    """Manages speech-to-text using Groq's Whisper API"""
    
    def __init__(self):
        self.groq_client = None
        self.sample_rate = 16000  # Standard for Whisper
        self.channels = 1
        self.is_recording = False
        self.recording_data = []
        self.system_audio_source = None
        self.recording_done_event = threading.Event()
        
        # Real-time transcription
        self.is_realtime_active = False
        self.realtime_thread = None
        self.chunk_duration = 5  # Process audio every 5 seconds
        self.transcription_callback = None
        

        
        # Set up system audio capture (what's playing on speakers/headphones)
        self.setup_system_audio_capture()
        self.init_groq_client()
        
        # Force refresh system audio source detection
        self.refresh_system_audio_source()
    
    def setup_system_audio_capture(self):
        """Set up system audio capture from monitor sources"""
        try:
            logging.info("Setting up system audio capture...")
            
            # Try to find and use system audio monitor sources
            try:
                import subprocess
                result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                      capture_output=True, text=True, check=True)
                sources = result.stdout.strip().split('\n')
                
                # Look for monitor sources (these capture system audio output)
                monitor_sources = []
                for source_line in sources:
                    if source_line and '.monitor' in source_line:
                        parts = source_line.split()
                        if len(parts) >= 2:
                            source_desc = parts[0]
                            source_name = parts[1]
                            monitor_sources.append((source_name, source_desc))
                            logging.info(f"Found monitor source: {source_name}")
                
                if monitor_sources:
                    # Prefer main audio output monitor over virtual sink monitor
                    for source_name, source_desc in monitor_sources:
                        if 'alsa_output' in source_name and 'analog-stereo.monitor' in source_name:
                            self.system_audio_source = source_name
                            logging.info(f"Using main audio output monitor: {source_name}")
                            break
                    
                    # If no main output found, use the first monitor source
                    if not self.system_audio_source and monitor_sources:
                        self.system_audio_source = monitor_sources[0][0]
                        logging.info(f"Using monitor source: {self.system_audio_source}")
                        
                    logging.info(f"✅ System audio capture configured with: {self.system_audio_source}")
                    logging.info("This will capture audio playing on your system (speakers/headphones)")
                else:
                    logging.warning("No monitor sources found")
                    
            except Exception as e:
                logging.error(f"Error getting PulseAudio sources: {e}")
            
            # Don't configure sounddevice here - we'll use parec directly
            if not self.system_audio_source:
                logging.warning("No system audio monitor found - will try to find one during recording")
                
        except Exception as e:
            logging.error(f"Error setting up system audio capture: {e}")
    
    def init_groq_client(self):
        """Initialize Groq client"""
        try:
            if not GROQ_AVAILABLE:
                logging.error("Groq package not available")
                return False
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logging.error("GROQ_API_KEY not found in environment")
                return False
            
            # Debug: Show first and last few characters of API key
            api_key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else api_key
            logging.info(f"🔑 Using Groq STT API key: {api_key_preview}")
            
            self.groq_client = Groq(api_key=api_key)
            logging.info("✅ Groq API loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            return False
    
    def refresh_system_audio_source(self):
        """Refresh and update system audio source detection"""
        try:
            logging.info("🔄 Refreshing system audio source detection...")
            
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True, check=True)
            sources = result.stdout.strip().split('\n')
            
            # Look for the best monitor source
            best_monitor = None
            fallback_monitor = None
            
            for source_line in sources:
                if source_line and '.monitor' in source_line:
                    parts = source_line.split()
                    if len(parts) >= 2:
                        source_name = parts[1]
                        
                        # Prefer hardware audio outputs
                        if 'alsa_output' in source_name and 'analog-stereo.monitor' in source_name:
                            best_monitor = source_name
                            logging.info(f"🎧 Found best system audio monitor: {source_name}")
                            break
                        elif 'beet_virtual_sink.monitor' not in source_name:
                            # Use as fallback if no hardware monitor found
                            if not fallback_monitor:
                                fallback_monitor = source_name
            
            # Update the system audio source
            if best_monitor:
                self.system_audio_source = best_monitor
                logging.info(f"✅ Updated system audio source to: {best_monitor}")
            elif fallback_monitor:
                self.system_audio_source = fallback_monitor
                logging.info(f"⚠️ Using fallback system audio source: {fallback_monitor}")
            else:
                logging.warning("❌ No suitable system audio monitor found")
                
        except Exception as e:
            logging.error(f"Error refreshing system audio source: {e}")

    def start_recording(self) -> bool:
        """Start recording system audio (what's playing on speakers/headphones)"""
        try:
            if not SOUNDDEVICE_AVAILABLE:
                logging.error("SoundDevice not available for recording")
                return False
            
            if self.is_recording:
                logging.warning("Already recording")
                return False
            
            self.refresh_system_audio_source()
            
            try:
                if self.system_audio_source:
                    logging.info(f"🎵 System audio source: {self.system_audio_source}")
                    logging.info("🎧 Recording system audio (what's playing on your computer)")
                else:
                    logging.warning("⚠️ No system audio source configured - may not capture system audio properly")
                    logging.warning("💡 Try running the debug_audio_sources.py script to diagnose")
            except Exception as e:
                logging.error(f"Error getting device info: {e}")
            
            self.recording_data = []
            self.is_recording = True
            self.recording_done_event.clear()
            
            def record_audio():
                try:
                    logging.info("[DEBUG] Entered record_audio thread")
                    print("[DEBUG] Entered record_audio thread")
                    duration = 30  # Maximum 30 seconds
                    logging.info(f"🎤 Recording system audio for up to {duration} seconds...")
                    logging.info("💡 Make sure audio is playing on your computer (music, meeting, video, etc.)")
                    logging.info(f"🔍 Thread started - is_recording: {self.is_recording}")
                    
                    if self.system_audio_source:
                        logging.info(f"📡 Using configured system audio source: {self.system_audio_source}")
                        # Use parec to capture from PulseAudio source
                        self.record_with_parec(duration)
                    else:
                        logging.info("⚠️ No system audio source configured, using sounddevice fallback")
                        # Fallback to sounddevice
                        self.record_with_sounddevice(duration)
                    
                    logging.info(f"🏁 Recording thread completed")
                except Exception as e:
                    logging.error(f"❌ Error during system audio recording: {e}")
                    import traceback
                    logging.error(f"❌ Full traceback: {traceback.format_exc()}")
                    print(f"[DEBUG] Exception in record_audio thread: {e}")
                    import traceback as tb; print(tb.format_exc())
                finally:
                    self.recording_done_event.set()
            
            logging.info("🚀 Starting recording thread...")
            threading.Thread(target=record_audio, daemon=True).start()
            return True
        except Exception as e:
            logging.error(f"Error starting system audio recording: {e}")
            return False
    
    def record_with_parec(self, duration: int):
        """Record system audio using PulseAudio's parec command"""
        try:
            logging.info("[DEBUG] Entered record_with_parec")
            print("[DEBUG] Entered record_with_parec")
            import tempfile
            import os
            import subprocess
            
            logging.info(f"🎙️ Starting record_with_parec for {duration} seconds")
            
            # Create temporary file for raw audio
            temp_raw = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
            temp_raw.close()
            logging.info(f"📁 Created temp file: {temp_raw.name}")
            
            # Find the correct monitor source for system audio
            monitor_source = None
            try:
                result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                      capture_output=True, text=True, check=True)
                sources = result.stdout.strip().split('\n')
                
                logging.info("🔍 Available audio sources:")
                for source_line in sources:
                    if source_line:
                        # Split by whitespace and take the second field (source name)
                        parts = source_line.split()
                        if len(parts) >= 2:
                            logging.info(f"  {parts[0]}: {parts[1]}")
                
                # Look for the main audio output monitor (not virtual sink monitor)
                # Prioritize hardware audio output monitors
                for source_line in sources:
                    if source_line and '.monitor' in source_line:
                        parts = source_line.split()  # Split by whitespace instead of tab
                        if len(parts) >= 2:
                            source_name = parts[1]
                            # Prefer hardware audio outputs over virtual sinks
                            if 'alsa_output' in source_name and 'analog-stereo.monitor' in source_name:
                                monitor_source = source_name
                                logging.info(f"✅ Found main audio monitor: {source_name}")
                                break
                            elif 'pulse_output' in source_name and '.monitor' in source_name:
                                monitor_source = source_name
                                logging.info(f"✅ Found pulse audio monitor: {source_name}")
                                break
                
                # If no hardware monitor found, use any available monitor (excluding our virtual sink)
                if not monitor_source:
                    for source_line in sources:
                        if source_line and '.monitor' in source_line:
                            parts = source_line.split()  # Split by whitespace instead of tab
                            if len(parts) >= 2:
                                source_name = parts[1]
                                # Skip our own virtual sink monitor
                                if 'beet_virtual_sink.monitor' not in source_name:
                                    monitor_source = source_name
                                    logging.info(f"✅ Using monitor: {source_name}")
                                    break
                                    
            except Exception as e:
                logging.error(f"❌ Error finding monitor source: {e}")
            
            if monitor_source:
                logging.info(f"🎧 Using parec to capture system audio from: {monitor_source}")
                print(f"[DEBUG] Using monitor source: {monitor_source}")
                
                # Use parec with explicit monitor source
                parec_cmd = [
                    'parec',
                    '--device', monitor_source,
                    '--format', 's16le',
                    '--rate', str(self.sample_rate),
                    '--channels', str(self.channels),
                    temp_raw.name
                ]
                logging.info(f"[DEBUG] parec command: {' '.join(parec_cmd)}")
                print(f"[DEBUG] parec command: {' '.join(parec_cmd)}")
            else:
                logging.error("❌ No suitable monitor source found for system audio capture!")
                logging.error("This means we cannot capture what's playing on your speakers/headphones")
                
                # Don't fallback to default - it won't work for system audio
                self.recording_data = []
                return
            
            logging.info(f"🚀 Starting parec: {' '.join(parec_cmd)}")
            logging.info(f"💡 Make sure audio is playing on your computer now!")
            
            # Start parec process
            process = subprocess.Popen(parec_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            logging.info(f"🎵 parec process started with PID: {process.pid}")
            
            # Wait for recording or stop signal
            start_time = time.time()
            logging.info(f"⏰ Recording started at {start_time}")
            
            while self.is_recording and (time.time() - start_time) < duration:
                if process.poll() is not None:  # Process ended
                    # Check if process ended with error
                    stderr_output = process.stderr.read().decode() if process.stderr else ""
                    if stderr_output:
                        logging.error(f"❌ parec stderr: {stderr_output}")
                    logging.warning(f"⚠️ parec process ended early after {time.time() - start_time:.2f} seconds")
                    break
                time.sleep(0.1)
            
            recording_duration = time.time() - start_time
            logging.info(f"⏱️ Recording loop completed after {recording_duration:.2f} seconds")
            logging.info(f"🔍 is_recording status: {self.is_recording}")
            
            # Stop the recording
            if process.poll() is None:
                logging.info("⏹️ Terminating parec process...")
                process.terminate()
                try:
                    process.wait(timeout=2)
                    logging.info("✅ parec process terminated cleanly")
                except subprocess.TimeoutExpired:
                    process.kill()
                    logging.warning("⚠️ Had to forcefully kill parec process")
            else:
                logging.info(f"📊 parec process already ended with return code: {process.returncode}")
            
            # Check if process ended successfully
            if process.returncode and process.returncode != 0:
                stderr_output = process.stderr.read().decode() if process.stderr else ""
                logging.error(f"❌ parec failed with return code {process.returncode}: {stderr_output}")
            else:
                logging.info("✅ parec process completed successfully")
            
            # --- DEBUG: Check file size and first bytes ---
            if os.path.exists(temp_raw.name):
                file_size = os.path.getsize(temp_raw.name)
                logging.info(f"[DEBUG] Raw audio file size: {file_size} bytes")
                print(f"[DEBUG] Raw audio file size: {file_size} bytes")
                with open(temp_raw.name, 'rb') as f:
                    first_bytes = f.read(32)
                logging.info(f"[DEBUG] First 32 bytes of raw file: {first_bytes}")
                print(f"[DEBUG] First 32 bytes of raw file: {first_bytes}")
            else:
                logging.error(f"[DEBUG] Raw audio file does not exist: {temp_raw.name}")
                print(f"[DEBUG] Raw audio file does not exist: {temp_raw.name}")
            # --- END DEBUG ---
            
            if os.path.exists(temp_raw.name) and os.path.getsize(temp_raw.name) > 0:
                # Read the raw audio data
                with open(temp_raw.name, 'rb') as f:
                    raw_data = f.read()
                
                # Convert to numpy array
                import numpy as np
                self.recording_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Debug: Check audio levels
                recorded_duration = time.time() - start_time
                max_val = np.max(np.abs(self.recording_data)) if len(self.recording_data) > 0 else 0
                rms_val = np.sqrt(np.mean(self.recording_data**2)) if len(self.recording_data) > 0 else 0
                
                logging.info(f"📊 Recorded {recorded_duration:.2f} seconds of system audio")
                logging.info(f"📊 Audio samples: {len(self.recording_data)}")
                logging.info(f"📊 Audio max value: {max_val}")
                logging.info(f"📊 Audio RMS: {rms_val:.2f}")
                logging.info(f"🔍 Recording data type: {type(self.recording_data)}")
                
                if max_val < 100:
                    logging.warning("⚠️ Very low system audio levels detected")
                    logging.warning("💡 Try playing audio louder or check if system audio is muted")
                    logging.warning("💡 Make sure audio is actually playing during recording")
                    logging.warning(f"💡 Audio source used: {monitor_source}")
                else:
                    logging.info("✅ Good system audio levels detected!")
                    logging.info("🎉 System audio capture successful - ready for transcription!")
                    logging.info(f"🎵 Audio captured from: {monitor_source}")
            else:
                file_size = os.path.getsize(temp_raw.name) if os.path.exists(temp_raw.name) else 0
                logging.error(f"❌ No audio data captured with parec (file size: {file_size} bytes)")
                logging.error("💡 This could mean:")
                logging.error("   - No audio was playing during recording")
                logging.error("   - System audio is muted")
                logging.error("   - Wrong audio source selected")
                logging.error("   - PulseAudio permission issue")
                self.recording_data = []
            
            # Clean up temp file
            try:
                os.unlink(temp_raw.name)
            except:
                pass
            
        except Exception as e:
            logging.error(f"Error with parec recording: {e}")
            # Fallback to sounddevice method
            self.record_with_sounddevice(duration)
    
    def record_with_sounddevice(self, duration: int):
        """Fallback recording method using sounddevice"""
        try:
            logging.info("Falling back to sounddevice recording...")
            
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            
            # Wait for recording or stop signal
            start_time = time.time()
            while self.is_recording and (time.time() - start_time) < duration:
                time.sleep(0.1)
            
            # Stop recording
            sd.stop()
            
            if self.is_recording:
                # Store only the recorded portion
                recorded_duration = time.time() - start_time
                recorded_samples = int(recorded_duration * self.sample_rate)
                self.recording_data = audio_data[:recorded_samples].flatten()
                
                # Debug: Check audio levels
                max_val = np.max(np.abs(self.recording_data)) if len(self.recording_data) > 0 else 0
                rms_val = np.sqrt(np.mean(self.recording_data**2)) if len(self.recording_data) > 0 else 0
                
                logging.info(f"📊 Recorded {recorded_duration:.2f} seconds with sounddevice")
                logging.info(f"📊 Audio samples: {len(self.recording_data)}")
                logging.info(f"📊 Audio max value: {max_val}")
                logging.info(f"📊 Audio RMS: {rms_val:.2f}")
                
                if max_val < 100:
                    logging.warning("⚠️ Very low audio levels detected")
                    logging.warning("💡 Check audio settings and make sure audio is playing")
                else:
                    logging.info("✅ Audio captured successfully!")
            
        except Exception as e:
            logging.error(f"Error with sounddevice recording: {e}")
            self.recording_data = []
    
    def stop_recording(self) -> bool:
        """Stop recording audio"""
        try:
            if not self.is_recording:
                logging.warning("Not currently recording")
                return False
            
            self.is_recording = False
            sd.stop()
            logging.info("Recording stopped")
            # Wait for recording thread to finish
            logging.info("Waiting for recording thread to finish...")
            self.recording_done_event.wait(timeout=5)  # Wait up to 5 seconds
            logging.info("Recording thread finished.")
            return True
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            return False
    
    def transcribe_recorded_audio(self) -> str:
        """Transcribe the recorded audio using Groq"""
        try:
            if not self.groq_client:
                logging.error("Groq client not initialized")
                return ""
            
            # Debug logging
            logging.info(f"🔍 Checking recording data... Type: {type(self.recording_data)}")
            if hasattr(self.recording_data, '__len__'):
                logging.info(f"🔍 Recording data length: {len(self.recording_data)}")
            else:
                logging.info(f"🔍 Recording data: {self.recording_data}")
            
            if self.recording_data is None or (hasattr(self.recording_data, '__len__') and len(self.recording_data) == 0):
                logging.error("❌ No recorded audio data available")
                logging.error("💡 This means the recording process didn't capture any audio data")
                logging.error("💡 Make sure audio is playing during recording and volume is up")
                return ""
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio data to wav file
            wavfile.write(temp_path, self.sample_rate, self.recording_data)
            
            # Check file size before sending to Groq
            file_size = os.path.getsize(temp_path)
            logging.info(f"🎧 Sending {file_size} bytes to Groq for transcription")
            
            if file_size == 0:
                logging.error("❌ Audio file is empty - cannot transcribe")
                return ""
            
            # Check minimum requirements from Groq docs
            if file_size < 1600:  # Very rough estimate for 0.01 seconds at 16kHz
                logging.warning("⚠️ Audio file might be too short (< 0.01 seconds)")
            
            # Transcribe using Groq (following their documentation)
            with open(temp_path, "rb") as file:
                logging.info("🚀 Calling Groq API for transcription...")
                transcription = self.groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",  # Fast, multilingual model (as per Groq docs)
                    response_format="text",
                    language="en",  # Can be made configurable
                    temperature=0.0  # Recommended by Groq docs for transcription
                )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            result_text = transcription.strip() if isinstance(transcription, str) else str(transcription)
            logging.info(f"Transcription completed: {len(result_text)} characters")
            return result_text
            
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""
    
    def transcribe_audio_file(self, file_path: str) -> str:
        """Transcribe audio from a file using Groq"""
        try:
            if not self.groq_client:
                logging.error("Groq client not initialized")
                return ""
            
            if not os.path.exists(file_path):
                logging.error(f"Audio file not found: {file_path}")
                return ""
            
            with open(file_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",  # Get detailed response
                    language="en"
                )
            
            # Extract text from response
            if hasattr(transcription, 'text'):
                result_text = transcription.text
            else:
                result_text = str(transcription)
            
            logging.info(f"File transcription completed: {len(result_text)} characters")
            return result_text
            
        except Exception as e:
            logging.error(f"Error transcribing audio file: {e}")
            return ""
    
    def start_realtime_transcription(self, callback_func=None) -> bool:
        """Start continuous real-time transcription"""
        try:
            if not self.groq_client:
                logging.error("Groq client not initialized")
                return False
            
            if self.is_realtime_active:
                logging.warning("Real-time transcription already active")
                return False
            
            self.transcription_callback = callback_func
            self.is_realtime_active = True
            
            # Start the real-time transcription thread
            self.realtime_thread = threading.Thread(target=self._realtime_transcription_loop, daemon=True)
            self.realtime_thread.start()
            
            logging.info("🎤 Real-time transcription started")
            return True
            
        except Exception as e:
            logging.error(f"Error starting real-time transcription: {e}")
            return False
    
    def stop_realtime_transcription(self) -> bool:
        """Stop continuous real-time transcription"""
        try:
            if not self.is_realtime_active:
                logging.warning("Real-time transcription not active")
                return False
            
            self.is_realtime_active = False
            
            # Wait for thread to finish
            if self.realtime_thread and self.realtime_thread.is_alive():
                self.realtime_thread.join(timeout=3)
            
            logging.info("⏹️ Real-time transcription stopped")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping real-time transcription: {e}")
            return False
    
    def _realtime_transcription_loop(self):
        """Main loop for real-time transcription"""
        try:
            logging.info("🔄 Real-time transcription loop started")
            while self.is_realtime_active:
                try:
                    # Record a chunk of audio
                    audio_chunk = self._record_audio_chunk(self.chunk_duration)
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        # Check if there's actual audio content (not just silence)
                        max_val = np.max(np.abs(audio_chunk)) if len(audio_chunk) > 0 else 0
                        if max_val > 500:  # Increased threshold to reduce false transcriptions
                            # Transcribe the chunk
                            transcription = self._transcribe_audio_chunk(audio_chunk)
                            if transcription and transcription.strip():
                                # Filter out common false transcriptions
                                false_transcriptions = [
                                    'thank you', 'thanks', 'you', 'hello', 'hi', 'bye', 'okay', 'ok', 
                                    'um', 'uh', 'ah', 'oh', 'hmm', 'yeah', 'yes', 'no', 'well', 'so',
                                    'and', 'the', 'a', 'an', 'i', 'is', 'are', 'was', 'were', 'be',
                                    'music', 'playing', 'sound', 'audio', 'noise'
                                ]
                                transcription_clean = transcription.strip().lower()
                                # Only emit callback if not a false transcription
                                if len(transcription_clean) >= 5 and transcription_clean not in false_transcriptions:
                                    if self.transcription_callback:
                                        self.transcription_callback(transcription_clean)
                                else:
                                    logging.info(f"[Filtered false transcription]: {transcription_clean}")
                                logging.info(f"📝 Transcribed: {transcription[:50]}...")
                        else:
                            # Silent period - just continue
                            pass
                    # Small delay to prevent overwhelming the API
                    time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Error in transcription loop iteration: {e}")
                    time.sleep(1)  # Wait before retrying
        except Exception as e:
            logging.error(f"Error in real-time transcription loop: {e}")
        finally:
            logging.info("🏁 Real-time transcription loop ended")
    
    def _record_audio_chunk(self, duration: float) -> np.ndarray:
        """Record a single chunk of audio for the specified duration"""
        try:
            self.refresh_system_audio_source()
            
            if not self.system_audio_source:
                logging.warning("No system audio source available")
                return np.array([])
            
            # Create temporary file for raw audio
            import tempfile
            import subprocess
            
            temp_raw = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
            temp_raw.close()
            
            # Use parec to capture audio chunk
            parec_cmd = [
                'timeout', str(duration + 0.5),  # Add small buffer
                'parec',
                '--device', self.system_audio_source,
                '--format', 's16le',
                '--rate', str(self.sample_rate),
                '--channels', str(self.channels),
                temp_raw.name
            ]
            
            # Run parec for the specified duration
            process = subprocess.run(parec_cmd, capture_output=True, timeout=duration + 2)
            
            # Read the audio data
            if os.path.exists(temp_raw.name) and os.path.getsize(temp_raw.name) > 0:
                with open(temp_raw.name, 'rb') as f:
                    raw_data = f.read()
                
                # Convert to numpy array
                audio_chunk = np.frombuffer(raw_data, dtype=np.int16)
                
                # Clean up temp file
                try:
                    os.unlink(temp_raw.name)
                except:
                    pass
                
                return audio_chunk
            else:
                # Clean up temp file
                try:
                    os.unlink(temp_raw.name)
                except:
                    pass
                return np.array([])
                
        except Exception as e:
            logging.error(f"Error recording audio chunk: {e}")
            return np.array([])
    
    def _transcribe_audio_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk"""
        try:
            if not self.groq_client:
                return ""
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save audio chunk to wav file
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, self.sample_rate, audio_chunk)
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            if file_size < 1600:  # Too small to transcribe
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return ""
            
            # Transcribe using Groq
            with open(temp_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    language="en",
                    temperature=0.0
                )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            result_text = transcription.strip() if isinstance(transcription, str) else str(transcription)
            return result_text
            
        except Exception as e:
            logging.error(f"Error transcribing audio chunk: {e}")
            return ""
    

    
    def capture_system_audio(self) -> bool:
        """Capture system audio output (advanced feature)"""
        try:
            # This would require capturing from the virtual sink monitor
            # For now, we'll focus on microphone input
            logging.info("System audio capture not yet implemented")
            return False
        except Exception as e:
            logging.error(f"Error capturing system audio: {e}")
            return False


class GroqTTSManager:
    """Manages Text-to-Speech using Groq's TTS API"""
    def __init__(self, groq_client, virtual_sink_name="beet_virtual_sink"):
        self.groq_client = groq_client
        self.virtual_sink_name = virtual_sink_name

    def speak_groq(self, text: str, voice: str = "Fritz-PlayAI", model: str = "playai-tts", response_format: str = "wav") -> bool:
        import tempfile
        import os
        try:
            if not self.groq_client:
                logging.error("Groq client not initialized for TTS")
                return False
            if not text.strip():
                logging.error("No text provided for Groq TTS")
                return False
            # Call Groq TTS API
            response = self.groq_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format
            )
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=f'.{response_format}', delete=False) as temp_file:
                temp_path = temp_file.name
                response.write_to_file(temp_path)
            # Play through virtual sink
            result = TTSManager(self.virtual_sink_name).play_audio_to_virtual_sink(temp_path)
            # Clean up
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            return result
        except Exception as e:
            logging.error(f"Error in Groq TTS: {e}")
            return False


class TTSManager:
    """Manages Text-to-Speech functionality and audio routing to virtual sink"""
    
    def __init__(self, virtual_sink_name="beet_virtual_sink"):
        self.virtual_sink_name = virtual_sink_name
        self.tts_engine = None
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
        self.init_tts_engine()
        # Add Groq TTS manager if Groq is available
        api_key = os.getenv('GROQ_API_KEY')
        self.groq_client = Groq(api_key=api_key) if GROQ_AVAILABLE and api_key else None
        self.groq_tts_manager = GroqTTSManager(self.groq_client, self.virtual_sink_name) if self.groq_client else None
    
    def init_tts_engine(self):
        """Initialize pyttsx3 TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Set properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume
        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
    
    def speak_offline(self, text: str) -> bool:
        """Speak text using offline TTS (pyttsx3) through virtual sink"""
        try:
            if not self.tts_engine:
                logging.error("TTS engine not initialized")
                return False
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save speech to file
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # Play through virtual sink
            result = self.play_audio_to_virtual_sink(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            logging.error(f"Error in offline TTS: {e}")
            return False
    
    def speak_online(self, text: str, language='en') -> bool:
        """Speak text using online TTS (gTTS) through virtual sink"""
        try:
            if not GTTS_AVAILABLE:
                logging.error("gTTS not available")
                return False
            
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                tts.save(temp_path)
            
            # Play through virtual sink
            result = self.play_audio_to_virtual_sink(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            logging.error(f"Error in online TTS: {e}")
            return False
    
    def play_audio_to_virtual_sink(self, audio_file: str) -> bool:
        """Play audio file through the virtual sink using ffmpeg/paplay"""
        try:
            # Try paplay first (more reliable for our use case)
            cmd = [
                'paplay', '--device', self.virtual_sink_name, audio_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully played audio through virtual sink using paplay")
                return True
            else:
                logging.error(f"paplay failed: {result.stderr}")
                
                # Fallback to ffmpeg
                cmd = [
                    'ffmpeg', '-i', audio_file, '-f', 'pulse', self.virtual_sink_name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logging.info(f"Successfully played audio through virtual sink using ffmpeg")
                    return True
                else:
                    logging.error(f"ffmpeg also failed: {result.stderr}")
                    return False
                
        except Exception as e:
            logging.error(f"Error playing audio to virtual sink: {e}")
            return False
    
    def get_available_voices(self) -> List[str]:
        """Get list of available TTS voices"""
        try:
            if self.tts_engine:
                voices = self.tts_engine.getProperty('voices')
                return [voice.name for voice in voices] if voices else []
            return []
        except Exception as e:
            logging.error(f"Error getting voices: {e}")
            return []
    
    def set_voice(self, voice_index: int):
        """Set TTS voice by index"""
        try:
            if self.tts_engine:
                voices = self.tts_engine.getProperty('voices')
                if voices and 0 <= voice_index < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_index].id)
                    return True
            return False
        except Exception as e:
            logging.error(f"Error setting voice: {e}")
            return False
    
    def set_rate(self, rate: int):
        """Set TTS speaking rate"""
        try:
            if self.tts_engine:
                self.tts_engine.setProperty('rate', rate)
                return True
            return False
        except Exception as e:
            logging.error(f"Error setting rate: {e}")
            return False
    
    def speak_groq(self, text: str, voice: str = "Fritz-PlayAI", model: str = "playai-tts", response_format: str = "wav") -> bool:
        if not self.groq_tts_manager:
            logging.error("Groq TTS manager not available")
            return False
        return self.groq_tts_manager.speak_groq(text, voice, model, response_format)


class PulseAudioManager:
    """Manages PulseAudio virtual devices"""
    
    def __init__(self):
        self.pulse = None
        self.virtual_sink_name = "beet_virtual_sink"
        self.virtual_source_name = "beet_virtual_mic"
        self.sink_module_id = None
        self.source_module_id = None
        self.connect_to_pulse()
    
    def connect_to_pulse(self):
        """Connect to PulseAudio"""
        try:
            self.pulse = pulsectl.Pulse('BEET')
            return True
        except Exception as e:
            logging.error(f"Failed to connect to PulseAudio: {e}")
            return False
    
    def create_virtual_microphone(self) -> bool:
        """Create virtual microphone using PulseAudio modules"""
        try:
            if not self.pulse:
                self.connect_to_pulse()
            
            # Remove existing virtual devices first
            self.remove_virtual_microphone()
            
            # Create null sink
            sink_cmd = [
                'pactl', 'load-module', 'module-null-sink',
                f'sink_name={self.virtual_sink_name}',
                'sink_properties=device.description=BEET-Virtual-Sink'
            ]
            
            result = subprocess.run(sink_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to create virtual sink: {result.stderr}")
                return False
            
            self.sink_module_id = result.stdout.strip()
            logging.info(f"Created virtual sink with module ID: {self.sink_module_id}")
            
            # Create remap source (virtual microphone)
            source_cmd = [
                'pactl', 'load-module', 'module-remap-source',
                f'master={self.virtual_sink_name}.monitor',
                f'source_name={self.virtual_source_name}',
                'source_properties=device.description=BEET-Virtual-Microphone'
            ]
            
            result = subprocess.run(source_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to create virtual source: {result.stderr}")
                # Clean up sink if source creation failed
                self.remove_virtual_microphone()
                return False
            
            self.source_module_id = result.stdout.strip()
            logging.info(f"Created virtual source with module ID: {self.source_module_id}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error creating virtual microphone: {e}")
            return False
    
    def remove_virtual_microphone(self) -> bool:
        """Remove virtual microphone modules"""
        try:
            success = True
            
            # Remove source module
            if self.source_module_id:
                result = subprocess.run(['pactl', 'unload-module', self.source_module_id], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logging.info(f"Removed virtual source module {self.source_module_id}")
                    self.source_module_id = None
                else:
                    logging.error(f"Failed to remove source module: {result.stderr}")
                    success = False
            
            # Remove sink module
            if self.sink_module_id:
                result = subprocess.run(['pactl', 'unload-module', self.sink_module_id], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    logging.info(f"Removed virtual sink module {self.sink_module_id}")
                    self.sink_module_id = None
                else:
                    logging.error(f"Failed to remove sink module: {result.stderr}")
                    success = False
            
            return success
            
        except Exception as e:
            logging.error(f"Error removing virtual microphone: {e}")
            return False
    
    def set_volume(self, volume: int):
        """Set virtual microphone volume (0-100)"""
        try:
            if not self.virtual_source_exists():
                return False
            
            # Convert percentage to PulseAudio volume (0-65536)
            pa_volume = int((volume / 100.0) * 65536)
            
            result = subprocess.run([
                'pactl', 'set-source-volume', 
                self.virtual_source_name, str(pa_volume)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Set volume to {volume}%")
                return True
            else:
                logging.error(f"Failed to set volume: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error setting volume: {e}")
            return False
    
    def virtual_source_exists(self) -> bool:
        """Check if virtual microphone source exists"""
        try:
            if not self.pulse:
                self.connect_to_pulse()
            
            if not self.pulse:
                return False
                
            sources = self.pulse.source_list()
            return any(source.name == self.virtual_source_name for source in sources)
        except Exception as e:
            logging.error(f"Error checking virtual source: {e}")
            return False
    
    def cleanup_all_virtual_devices(self) -> bool:
        """Remove all virtual audio devices (null sinks and remap sources)"""
        try:
            logging.info("🧹 Starting cleanup of all virtual audio devices...")
            
            # Get list of all loaded modules
            result = subprocess.run(['pactl', 'list', 'modules', 'short'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error("Failed to get module list")
                return False
            
            modules = result.stdout.strip().split('\n')
            removed_count = 0
            
            for module_line in modules:
                if module_line:
                    parts = module_line.split('\t')
                    if len(parts) >= 2:
                        module_id = parts[0]
                        module_type = parts[1]
                        
                        # Remove virtual audio modules
                        if module_type in ['module-null-sink', 'module-remap-source']:
                            try:
                                unload_result = subprocess.run(
                                    ['pactl', 'unload-module', module_id], 
                                    capture_output=True, text=True
                                )
                                
                                if unload_result.returncode == 0:
                                    logging.info(f"✅ Removed {module_type} (ID: {module_id})")
                                    removed_count += 1
                                else:
                                    logging.warning(f"⚠️ Failed to remove {module_type} (ID: {module_id}): {unload_result.stderr}")
                                    
                            except Exception as e:
                                logging.error(f"❌ Error removing module {module_id}: {e}")
            
            # Reset our internal state
            self.sink_module_id = None
            self.source_module_id = None
            
            if removed_count > 0:
                logging.info(f"🎉 Successfully removed {removed_count} virtual audio device(s)")
                return True
            else:
                logging.info("ℹ️ No virtual audio devices found to remove")
                return True
                
        except Exception as e:
            logging.error(f"❌ Error during virtual device cleanup: {e}")
            return False
    
    def get_audio_devices(self) -> Dict[str, List]:
        """Get all audio sources and sinks"""
        try:
            if not self.pulse:
                self.connect_to_pulse()
            
            sources = []
            sinks = []
            
            if self.pulse:
                for source in self.pulse.source_list():
                    sources.append({
                        'name': source.name,
                        'description': source.description,
                        'index': source.index
                    })
                
                for sink in self.pulse.sink_list():
                    sinks.append({
                        'name': sink.name,
                        'description': sink.description,
                        'index': sink.index
                    })
            
            return {'sources': sources, 'sinks': sinks}
            
        except Exception as e:
            logging.error(f"Error getting audio devices: {e}")
            return {'sources': [], 'sinks': []}


class LogHandler(logging.Handler):
    """Custom logging handler that emits signals for GUI"""
    
    def __init__(self):
        super().__init__()
        self.log_signal = None
    
    def set_log_signal(self, signal):
        """Set the log signal for emitting messages"""
        self.log_signal = signal
    
    def emit(self, record):
        if self.log_signal:
            msg = self.format(record)
            self.log_signal.emit(msg)


class BEETMainWindow(QMainWindow):
    """Main window for BEET Virtual Microphone Application"""
    
    log_signal = pyqtSignal(str)
    transcription_signal = pyqtSignal(str, int)  # text, length
    ai_response_signal = pyqtSignal(str, bool)  # response_chunk, is_complete
    
    def __init__(self):
        super().__init__()
        self.pulse_manager = PulseAudioManager()
        self.tts_manager = TTSManager(self.pulse_manager.virtual_sink_name)
        self.stt_manager = GroqSTTManager()
        self.text_gen_manager = GroqTextGenerationManager()
        # Initialize states first
        self.meeting_join_worker = None
        self.is_realtime_transcription_active = False
        self.auto_generate_responses = True  # Auto-enabled by default
        self.current_ai_response = ""
        self.ai_is_speaking = False  # Flag to prevent feedback loops
        self.last_ai_response_text = ""  # Track last AI response to avoid responding to it
        self.init_logging()
        self.init_ui()
        self.setup_timer()
        self.transcription_signal.connect(self.handle_transcription_result)
        self.ai_response_signal.connect(self.handle_ai_response)
        # self.init_ai_status()  # Removed: no longer needed after always-auto AI response
        # Check PulseAudio availability
        if not self.pulse_manager.pulse:
            QMessageBox.critical(
                self, "PulseAudio Error",
                "Could not connect to PulseAudio. Please ensure PulseAudio is running."
            )
        # Automatically create virtual mic if not present
        if not self.pulse_manager.virtual_source_exists():
            self.create_virtual_mic()
        else:
            self.create_btn.setEnabled(False)
            self.remove_btn.setEnabled(True)
            self.volume_slider.setEnabled(True)
            self.offline_tts_btn.setEnabled(True)
            self.online_tts_btn.setEnabled(True)
            self.status_label.setText("Status: Virtual microphone active")
            self.tts_status_label.setText("Status: TTS ready - Virtual microphone active")
        
        # Automatically start real-time transcription if Groq client is available
        if self.stt_manager.groq_client:
            success = self.stt_manager.start_realtime_transcription(callback_func=self.on_realtime_transcription)
            if success:
                self.is_realtime_transcription_active = True
                self.toggle_realtime_btn.setText("⏹️ Stop Real-time Transcription")
                self.toggle_realtime_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #d32f2f;
                        color: white;
                        padding: 10px 18px;
                        border-radius: 6px;
                        font-weight: bold;
                        font-size: 11px;
                        min-height: 20px;
                    }
                    QPushButton:hover {
                        background-color: #b71c1c;
                        border: 1px solid #ff5252;
                    }
                """)
                self.realtime_status.setText("🔴 Real-time transcription: ACTIVE")
                self.realtime_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.stt_status_label.setText("Status: 🎤 Listening continuously...")
                self.status_bar.showMessage("Real-time transcription started - listening for audio...")
            else:
                self.stt_status_label.setText("Status: Failed to start real-time transcription")
                self.status_bar.showMessage("Failed to start real-time transcription")
    
    def init_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Add custom handler for GUI
        self.log_handler = LogHandler()
        self.log_handler.set_log_signal(self.log_signal)
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        )
        logging.getLogger().addHandler(self.log_handler)
        
        self.log_signal.connect(self.append_log)
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("BEET - Virtual Microphone")
        self.setGeometry(100, 100, 800, 600)
        
        # Apply modern dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #00d4ff;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border: 1px solid #00d4ff;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                background: #2d2d2d;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00d4ff, stop: 1 #0078d4);
                border: 1px solid #404040;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #ffffff, stop: 1 #d0d0d0);
                border: 2px solid #404040;
                width: 20px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #00d4ff, stop: 1 #0078d4);
                border: 2px solid #00d4ff;
            }
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                selection-background-color: #0078d4;
            }
            QTextEdit:focus {
                border: 2px solid #00d4ff;
            }
            QListWidget {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 5px;
                color: #ffffff;
                alternate-background-color: #353535;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #404040;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #404040;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 6px;
                padding: 5px 10px;
                color: #ffffff;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 2px solid #00d4ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                selection-background-color: #0078d4;
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 6px;
                background-color: #2d2d2d;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00d4ff, stop: 1 #0078d4);
                border-radius: 4px;
                margin: 2px;
            }
            QStatusBar {
                background-color: #252525;
                border-top: 1px solid #404040;
                color: #ffffff;
                font-size: 11px;
            }
            QSplitter::handle {
                background-color: #404040;
                width: 3px;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #00d4ff;
            }
            QFrame {
                background-color: #1e1e1e;
                border: none;
            }
            QTabWidget::pane {
                border: 2px solid #404040;
                border-radius: 6px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #505050;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Device list and logs
        right_panel = self.create_info_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 400])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        logging.info("BEET Virtual Microphone Application Started")
    
    def create_control_panel(self) -> QWidget:
        """Create the main control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Virtual Microphone Control Group
        mic_group = QGroupBox("Virtual Microphone Control")
        mic_layout = QVBoxLayout(mic_group)
        
        # Create/Remove buttons
        button_layout = QHBoxLayout()
        
        self.create_btn = QPushButton("🎤 Create Virtual Mic")
        self.create_btn.clicked.connect(self.create_virtual_mic)
        button_layout.addWidget(self.create_btn)
        
        self.remove_btn = QPushButton("❌ Remove Virtual Mic")
        self.remove_btn.clicked.connect(self.remove_virtual_mic)
        self.remove_btn.setEnabled(False)
        button_layout.addWidget(self.remove_btn)
        
        mic_layout.addLayout(button_layout)
        
        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setEnabled(False)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("50%")
        self.volume_label.setMinimumWidth(40)
        volume_layout.addWidget(self.volume_label)
        
        mic_layout.addLayout(volume_layout)
        
        # Status
        self.status_label = QLabel("Status: Virtual microphone not created")
        mic_layout.addWidget(self.status_label)
        
        layout.addWidget(mic_group)
        
        # Device Management Group
        device_group = QGroupBox("Device Management")
        device_layout = QVBoxLayout(device_group)
        
        # Device control buttons
        device_buttons_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh Device List")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        device_buttons_layout.addWidget(self.refresh_btn)
        
        self.cleanup_btn = QPushButton("🧹 Cleanup All Virtual Mics")
        self.cleanup_btn.clicked.connect(self.cleanup_all_virtual_mics)
        self.cleanup_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                padding: 8px 14px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
                border: 1px solid #ff8a65;
            }
            QPushButton:pressed {
                background-color: #d84315;
            }
        """)
        device_buttons_layout.addWidget(self.cleanup_btn)
        
        device_layout.addLayout(device_buttons_layout)
        
        layout.addWidget(device_group)
        
        # TTS Control Group
        tts_group = QGroupBox("Text-to-Speech Control")
        tts_layout = QVBoxLayout(tts_group)
        
        # Text input
        self.tts_text = QTextEdit()
        self.tts_text.setPlaceholderText("Enter text to speak through virtual microphone...")
        self.tts_text.setMaximumHeight(100)
        tts_layout.addWidget(QLabel("Text to Speak:"))
        tts_layout.addWidget(self.tts_text)
        
        # TTS options
        options_layout = QHBoxLayout()
        
        # Offline TTS button
        self.offline_tts_btn = QPushButton("🗣️ Speak (Offline)")
        self.offline_tts_btn.clicked.connect(self.speak_offline)
        self.offline_tts_btn.setEnabled(False)
        options_layout.addWidget(self.offline_tts_btn)
        
        # Online TTS button
        self.online_tts_btn = QPushButton("🌐 Speak (Online)")
        self.online_tts_btn.clicked.connect(self.speak_online)
        self.online_tts_btn.setEnabled(False)
        options_layout.addWidget(self.online_tts_btn)
        
        tts_layout.addLayout(options_layout)
        
        # TTS settings
        settings_layout = QHBoxLayout()
        
        # Speech rate
        settings_layout.addWidget(QLabel("Speed:"))
        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(50, 300)
        self.rate_slider.setValue(150)
        self.rate_slider.valueChanged.connect(self.on_rate_changed)
        settings_layout.addWidget(self.rate_slider)
        
        self.rate_label = QLabel("150")
        self.rate_label.setMinimumWidth(30)
        settings_layout.addWidget(self.rate_label)
        
        tts_layout.addLayout(settings_layout)
        
        # TTS status
        self.tts_status_label = QLabel("Status: TTS ready (requires virtual microphone)")
        tts_layout.addWidget(self.tts_status_label)
        
        layout.addWidget(tts_group)
        

        

        
        # Speech-to-Text Control Group
        stt_group = QGroupBox("Speech-to-Text Control (Groq)")
        stt_layout = QVBoxLayout(stt_group)
        
        # Real-time transcription controls
        record_layout = QHBoxLayout()
        
        self.toggle_realtime_btn = QPushButton("🎤 Start Real-time Transcription")
        self.toggle_realtime_btn.clicked.connect(self.toggle_realtime_transcription)
        self.toggle_realtime_btn.setEnabled(True)
        record_layout.addWidget(self.toggle_realtime_btn)
        
        # Options layout
        options_layout = QVBoxLayout()
        
        # Auto-clear toggle
        self.auto_clear_checkbox = QCheckBox("Auto-clear old transcriptions")
        self.auto_clear_checkbox.setChecked(True)
        self.auto_clear_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #404040;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 2px solid #00d4ff;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #00d4ff;
            }
        """)
        options_layout.addWidget(self.auto_clear_checkbox)
        
        # Independent AI responses toggle
        self.independent_ai_checkbox = QCheckBox("Independent AI responses (no conversation context)")
        self.independent_ai_checkbox.setChecked(True)  # Default to independent responses
        self.independent_ai_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #404040;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 2px solid #00d4ff;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #00d4ff;
            }
        """)
        options_layout.addWidget(self.independent_ai_checkbox)
        

        
        record_layout.addLayout(options_layout)
        
        stt_layout.addLayout(record_layout)
        
        # Real-time status indicator
        self.realtime_status = QLabel("⚫ Real-time transcription: OFF")
        self.realtime_status.setStyleSheet("color: #808080; font-weight: bold;")
        stt_layout.addWidget(self.realtime_status)
        
        # File transcription
        file_layout = QHBoxLayout()
        
        self.transcribe_file_btn = QPushButton("📁 Transcribe Audio File")
        self.transcribe_file_btn.clicked.connect(self.transcribe_file)
        file_layout.addWidget(self.transcribe_file_btn)
        
        stt_layout.addLayout(file_layout)
        
        # STT status
        self.stt_status_label = QLabel("Status: Ready for speech-to-text")
        stt_layout.addWidget(self.stt_status_label)
        
        layout.addWidget(stt_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_info_panel(self) -> QWidget:
        """Create the information panel with device list and logs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Device List Group
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout(device_group)
        
        self.device_list = QListWidget()
        device_layout.addWidget(self.device_list)
        
        layout.addWidget(device_group)
        
        # Transcription Results Group
        transcription_group = QGroupBox("Transcription Results")
        transcription_layout = QVBoxLayout(transcription_group)
        
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(False)  # Allow editing for corrections
        self.transcription_text.setPlaceholderText("Transcribed text will appear here...")
        self.transcription_text.setMaximumHeight(120)
        self.transcription_text.setFont(QFont("Arial", 10))
        transcription_layout.addWidget(self.transcription_text)
        
        # Transcription controls
        trans_controls_layout = QHBoxLayout()
        
        self.clear_transcription_btn = QPushButton("🗑️ Clear")
        self.clear_transcription_btn.clicked.connect(self.clear_transcription)
        trans_controls_layout.addWidget(self.clear_transcription_btn)
        
        self.copy_transcription_btn = QPushButton("📋 Copy to TTS")
        self.copy_transcription_btn.clicked.connect(self.copy_to_tts)
        trans_controls_layout.addWidget(self.copy_transcription_btn)
        
        transcription_layout.addLayout(trans_controls_layout)
        
        layout.addWidget(transcription_group)
        
        # AI Response Group
        ai_group = QGroupBox("🤖 AI Response Generation (Groq LLaMA)")
        ai_layout = QVBoxLayout(ai_group)
        
        # AI response display
        self.ai_response_text = QTextEdit()
        self.ai_response_text.setReadOnly(True)
        self.ai_response_text.setPlaceholderText("AI responses will appear here...")
        self.ai_response_text.setMaximumHeight(120)
        self.ai_response_text.setFont(QFont("Arial", 10))
        ai_layout.addWidget(self.ai_response_text)
        
        # AI controls
        ai_controls_layout = QHBoxLayout()
        
        self.clear_ai_response_btn = QPushButton("🗑️ Clear Response")
        self.clear_ai_response_btn.clicked.connect(self.clear_ai_response)
        self.clear_ai_response_btn.setToolTip("Clear AI response text")
        ai_controls_layout.addWidget(self.clear_ai_response_btn)
        
        self.copy_ai_to_tts_btn = QPushButton("📋 Copy to TTS")
        self.copy_ai_to_tts_btn.clicked.connect(self.copy_ai_to_tts)
        self.copy_ai_to_tts_btn.setToolTip("Copy AI response to text-to-speech input")
        ai_controls_layout.addWidget(self.copy_ai_to_tts_btn)
        
        self.clear_ai_conversation_btn = QPushButton("🗑️ Clear History")
        self.clear_ai_conversation_btn.clicked.connect(self.clear_ai_conversation)
        self.clear_ai_conversation_btn.setToolTip("Clear AI conversation history")
        ai_controls_layout.addWidget(self.clear_ai_conversation_btn)
        
        self.refresh_api_key_btn = QPushButton("🔄 Refresh API Key")
        self.refresh_api_key_btn.clicked.connect(self.refresh_api_keys)
        self.refresh_api_key_btn.setToolTip("Reload API key from environment (useful after changing API key)")
        ai_controls_layout.addWidget(self.refresh_api_key_btn)
        
        ai_layout.addLayout(ai_controls_layout)
        
        # AI status
        self.ai_status_label = QLabel("Status: Ready for AI response generation")
        ai_layout.addWidget(self.ai_status_label)
        
        layout.addWidget(ai_group)
        
        # Log Group
        log_group = QGroupBox("Application Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)  # Reduced to make room for transcription
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
    
    def setup_timer(self):
        """Setup timer for periodic updates"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(2000)  # Update every 2 seconds
    
    def create_virtual_mic(self):
        """Create virtual microphone"""
        self.status_bar.showMessage("Creating virtual microphone...")
        
        if self.pulse_manager.create_virtual_microphone():
            self.create_btn.setEnabled(False)
            self.remove_btn.setEnabled(True)
            self.volume_slider.setEnabled(True)
            self.offline_tts_btn.setEnabled(True)
            self.online_tts_btn.setEnabled(True)
            self.status_label.setText("Status: Virtual microphone active")
            self.tts_status_label.setText("Status: TTS ready - Virtual microphone active")
            self.status_bar.showMessage("Virtual microphone created successfully")
            
            # Set initial volume
            self.pulse_manager.set_volume(self.volume_slider.value())
            
            QMessageBox.information(
                self, "Success",
                "Virtual microphone created successfully!\n\n"
                "You can now use 'BEET Virtual Microphone' as an audio input "
                "in other applications."
            )
        else:
            QMessageBox.critical(
                self, "Error",
                "Failed to create virtual microphone.\n\n"
                "Please check that PulseAudio is running and you have "
                "the necessary permissions."
            )
            self.status_bar.showMessage("Failed to create virtual microphone")
    
    def remove_virtual_mic(self):
        """Remove virtual microphone"""
        self.status_bar.showMessage("Removing virtual microphone...")
        
        if self.pulse_manager.remove_virtual_microphone():
            self.create_btn.setEnabled(True)
            self.remove_btn.setEnabled(False)
            self.volume_slider.setEnabled(False)
            self.offline_tts_btn.setEnabled(False)
            self.online_tts_btn.setEnabled(False)
            self.status_label.setText("Status: Virtual microphone not created")
            self.tts_status_label.setText("Status: TTS ready (requires virtual microphone)")
            self.status_bar.showMessage("Virtual microphone removed")
        else:
            QMessageBox.warning(
                self, "Warning",
                "Some components may not have been removed completely.\n"
                "Check the log for details."
            )
            self.status_bar.showMessage("Virtual microphone removal completed with warnings")
    
    def on_volume_changed(self, value):
        """Handle volume slider change"""
        self.volume_label.setText(f"{value}%")
        
        if self.volume_slider.isEnabled():
            self.pulse_manager.set_volume(value)
    
    def on_rate_changed(self, value):
        """Handle TTS rate slider change"""
        self.rate_label.setText(str(value))
        self.tts_manager.set_rate(value)
    
    def speak_offline(self):
        """Speak text using offline TTS"""
        text = self.tts_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text to speak.")
            return
        
        if not self.pulse_manager.virtual_source_exists():
            QMessageBox.warning(
                self, "Virtual Microphone Required",
                "Please create a virtual microphone first."
            )
            return
        
        self.offline_tts_btn.setEnabled(False)
        self.online_tts_btn.setEnabled(False)
        self.tts_status_label.setText("Status: Speaking (offline)...")
        self.status_bar.showMessage("Speaking text through virtual microphone...")
        
        # Run TTS in a separate thread to avoid blocking UI
        def tts_worker():
            try:
                success = self.tts_manager.speak_offline(text)
                if success:
                    self.tts_status_label.setText("Status: Speech completed successfully")
                    self.status_bar.showMessage("Speech completed")
                else:
                    self.tts_status_label.setText("Status: Speech failed")
                    self.status_bar.showMessage("Speech failed")
            except Exception as e:
                logging.error(f"TTS error: {e}")
                self.tts_status_label.setText("Status: Speech error")
                self.status_bar.showMessage("Speech error")
            finally:
                self.offline_tts_btn.setEnabled(True)
                self.online_tts_btn.setEnabled(True)
        
        threading.Thread(target=tts_worker, daemon=True).start()
    
    def speak_online(self):
        """Speak text using online TTS"""
        text = self.tts_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text to speak.")
            return
        
        if not self.pulse_manager.virtual_source_exists():
            QMessageBox.warning(
                self, "Virtual Microphone Required",
                "Please create a virtual microphone first."
            )
            return
        
        self.offline_tts_btn.setEnabled(False)
        self.online_tts_btn.setEnabled(False)
        self.tts_status_label.setText("Status: Speaking (online)...")
        self.status_bar.showMessage("Speaking text through virtual microphone (online)...")
        
        # Run TTS in a separate thread to avoid blocking UI
        def tts_worker():
            try:
                success = self.tts_manager.speak_online(text)
                if success:
                    self.tts_status_label.setText("Status: Speech completed successfully")
                    self.status_bar.showMessage("Speech completed")
                else:
                    self.tts_status_label.setText("Status: Speech failed")
                    self.status_bar.showMessage("Speech failed")
            except Exception as e:
                logging.error(f"TTS error: {e}")
                self.tts_status_label.setText("Status: Speech error")
                self.status_bar.showMessage("Speech error")
            finally:
                self.offline_tts_btn.setEnabled(True)
                self.online_tts_btn.setEnabled(True)
        
        threading.Thread(target=tts_worker, daemon=True).start()
    
    def toggle_realtime_transcription(self):
        """Toggle real-time transcription on/off"""
        if not self.stt_manager.groq_client:
            QMessageBox.warning(
                self, "Groq API Key Required",
                "Please set your GROQ_API_KEY environment variable.\n\n"
                "You can get a free API key at: https://console.groq.com/"
            )
            return
        
        if not self.is_realtime_transcription_active:
            # Start real-time transcription
            success = self.stt_manager.start_realtime_transcription(
                callback_func=self.on_realtime_transcription
            )
            
            if success:
                self.is_realtime_transcription_active = True
                self.toggle_realtime_btn.setText("⏹️ Stop Real-time Transcription")
                self.toggle_realtime_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #d32f2f;
                        color: white;
                        padding: 10px 18px;
                        border-radius: 6px;
                        font-weight: bold;
                        font-size: 11px;
                        min-height: 20px;
                    }
                    QPushButton:hover {
                        background-color: #b71c1c;
                        border: 1px solid #ff5252;
                    }
                """)
                self.realtime_status.setText("🔴 Real-time transcription: ACTIVE")
                self.realtime_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.stt_status_label.setText("Status: 🎤 Listening continuously...")
                self.status_bar.showMessage("Real-time transcription started - listening for audio...")
                
                # Clear transcription area if auto-clear is enabled
                if self.auto_clear_checkbox.isChecked():
                    self.transcription_text.clear()
            else:
                self.stt_status_label.setText("Status: Failed to start real-time transcription")
                self.status_bar.showMessage("Failed to start real-time transcription")
        else:
            # Stop real-time transcription
            success = self.stt_manager.stop_realtime_transcription()
            
            if success:
                self.is_realtime_transcription_active = False
                self.toggle_realtime_btn.setText("🎤 Start Real-time Transcription")
                self.toggle_realtime_btn.setStyleSheet("")  # Reset to default style
                self.realtime_status.setText("⚫ Real-time transcription: OFF")
                self.realtime_status.setStyleSheet("color: #808080; font-weight: bold;")
                self.stt_status_label.setText("Status: Real-time transcription stopped")
                self.status_bar.showMessage("Real-time transcription stopped")
            else:
                self.stt_status_label.setText("Status: Error stopping real-time transcription")
                self.status_bar.showMessage("Error stopping real-time transcription")
    
    def on_realtime_transcription(self, transcription_text: str):
        """Handle real-time transcription results"""
        try:
            # This method is called from the transcription thread
            # We need to emit a signal to update the UI in the main thread
            self.transcription_signal.emit(transcription_text, len(transcription_text))
        except Exception as e:
            logging.error(f"Error handling real-time transcription: {e}")
    
    def transcribe_file(self):
        """Transcribe an audio file"""
        if not self.stt_manager.groq_client:
            QMessageBox.warning(
                self, "Groq API Key Required",
                "Please set your GROQ_API_KEY environment variable.\n\n"
                "You can get a free API key at: https://console.groq.com/"
            )
            return
        
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File for Transcription",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.ogg *.flac *.mp4);;All Files (*)"
        )
        
        if file_path:
            self.stt_status_label.setText("Status: Transcribing file...")
            self.status_bar.showMessage(f"Transcribing {os.path.basename(file_path)}...")
            
            def transcribe_file_worker():
                try:
                    transcription = self.stt_manager.transcribe_audio_file(file_path)
                    if transcription:
                        self.transcription_text.setPlainText(transcription)
                        self.stt_status_label.setText(f"Status: File transcribed ({len(transcription)} chars)")
                        self.status_bar.showMessage("File transcription completed")
                    else:
                        self.stt_status_label.setText("Status: File transcription failed")
                        self.status_bar.showMessage("File transcription failed")
                except Exception as e:
                    logging.error(f"File transcription error: {e}")
                    self.stt_status_label.setText("Status: File transcription error")
                    self.status_bar.showMessage("File transcription error")
            
            threading.Thread(target=transcribe_file_worker, daemon=True).start()
    
    def clear_transcription(self):
        """Clear the transcription text"""
        self.transcription_text.clear()
        self.stt_status_label.setText("Status: Ready for speech-to-text")
    
    def copy_to_tts(self):
        """Copy transcription text to TTS input"""
        transcription = self.transcription_text.toPlainText().strip()
        if transcription:
            self.tts_text.setPlainText(transcription)
            self.status_bar.showMessage("Transcription copied to TTS input")
        else:
            QMessageBox.information(self, "No Text", "No transcription text to copy.")
    

    
    def refresh_devices(self):
        """Refresh audio device list"""
        self.status_bar.showMessage("Refreshing device list...")
        
        devices = self.pulse_manager.get_audio_devices()
        self.device_list.clear()
        
        # Add sources
        self.device_list.addItem("=== AUDIO SOURCES (Microphones) ===")
        for source in devices['sources']:
            item_text = f"🎤 {source['description']} ({source['name']})"
            self.device_list.addItem(item_text)
        
        self.device_list.addItem("")
        
        # Add sinks
        self.device_list.addItem("=== AUDIO SINKS (Speakers) ===")
        for sink in devices['sinks']:
            item_text = f"🔊 {sink['description']} ({sink['name']})"
            self.device_list.addItem(item_text)
        
        self.status_bar.showMessage("Device list refreshed")
    
    def cleanup_all_virtual_mics(self):
        """Remove all virtual microphones and audio devices"""
        reply = QMessageBox.question(
            self, "Cleanup Virtual Devices",
            "This will remove ALL virtual audio devices (null sinks and remap sources) from PulseAudio.\n\n"
            "This includes:\n"
            "• BEET virtual microphones\n"
            "• Any other virtual audio devices\n"
            "• Third-party virtual audio tools\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.status_bar.showMessage("Cleaning up virtual audio devices...")
            
            success = self.pulse_manager.cleanup_all_virtual_devices()
            
            if success:
                # Update UI state
                self.create_btn.setEnabled(True)
                self.remove_btn.setEnabled(False)
                self.volume_slider.setEnabled(False)
                self.offline_tts_btn.setEnabled(False)
                self.online_tts_btn.setEnabled(False)
                self.status_label.setText("Status: Virtual microphone not created")
                self.tts_status_label.setText("Status: TTS ready (requires virtual microphone)")
                
                # Refresh device list
                self.refresh_devices()
                
                self.status_bar.showMessage("Virtual device cleanup completed")
                QMessageBox.information(
                    self, "Cleanup Complete",
                    "All virtual audio devices have been removed successfully.\n\n"
                    "You can now create a new virtual microphone if needed."
                )
            else:
                self.status_bar.showMessage("Virtual device cleanup failed")
                QMessageBox.warning(
                    self, "Cleanup Failed",
                    "Some virtual devices could not be removed.\n\n"
                    "Check the application log for details."
                )
    
    def update_status(self):
        """Update application status"""
        if self.pulse_manager.virtual_source_exists():
            if not self.remove_btn.isEnabled():
                # Virtual mic exists but GUI thinks it doesn't - sync state
                self.create_btn.setEnabled(False)
                self.remove_btn.setEnabled(True)
                self.volume_slider.setEnabled(True)
                self.status_label.setText("Status: Virtual microphone active")
        else:
            if self.remove_btn.isEnabled():
                # Virtual mic doesn't exist but GUI thinks it does - sync state
                self.create_btn.setEnabled(True)
                self.remove_btn.setEnabled(False)
                self.volume_slider.setEnabled(False)
                self.status_label.setText("Status: Virtual microphone not created")
    
    def append_log(self, message):
        """Append message to log"""
        self.log_text.append(message)
        
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
    
    def handle_transcription_result(self, transcription, length):
        if transcription:
            # Anti-feedback logic: Don't process transcriptions while AI is speaking
            if self.ai_is_speaking:
                logging.info(f"🔇 Ignoring transcription while AI is speaking: {transcription[:30]}...")
                return
            # Anti-feedback logic: Don't respond to AI's own speech
            if self.is_ai_own_speech(transcription):
                logging.info(f"🤖 Ignoring AI's own speech: {transcription[:30]}...")
                return
            if self.is_realtime_transcription_active:
                # For real-time mode, clear previous text and show only the latest transcription
                current_time = time.strftime("%H:%M:%S")
                # Clear previous text and show only the current transcription with timestamp
                new_text = f"[{current_time}] {transcription}"
                self.transcription_text.setPlainText(new_text)
                # Auto-scroll to bottom
                cursor = self.transcription_text.textCursor()
                cursor.movePosition(cursor.End)
                self.transcription_text.setTextCursor(cursor)
                # Update status for real-time mode
                self.stt_status_label.setText(f"Status: 🎤 Listening... (latest: {len(transcription)} chars)")
                # Send the latest statement to AI
                latest_statement = transcription.strip()
                logging.info(f"🤖 Auto-generating AI response for: {latest_statement[:50]}...")
                self.generate_ai_response(latest_statement)
            else:
                # For file transcription mode, replace content
                self.transcription_text.setPlainText(transcription)
                self.stt_status_label.setText(f"Status: Transcription completed ({length} chars)")
                self.status_bar.showMessage("Transcription completed successfully")
                # For file mode, send only the latest statement (last sentence)
                import re
                statements = re.split(r'(?<=[.!?])\s+', transcription.strip())
                latest_statement = statements[-1] if statements else transcription.strip()
                logging.info(f"🤖 Auto-generating AI response for: {latest_statement[:50]}...")
                self.generate_ai_response(latest_statement)
        else:
            if not self.is_realtime_transcription_active:
                self.transcription_text.clear()
                self.stt_status_label.setText("Status: Transcription failed")
                self.status_bar.showMessage("Transcription failed - check logs")
    
    def generate_ai_response(self, user_input: str):
        """Generate AI response to user input"""
        if not self.text_gen_manager.groq_client:
            self.ai_status_label.setText("Status: Groq API key required for AI responses")
            return
        
        if not user_input.strip():
            return
        
        self.ai_status_label.setText("Status: 🤖 Generating AI response...")
        
        def generate_worker():
            try:
                def streaming_callback(chunk, is_complete):
                    self.ai_response_signal.emit(chunk, is_complete)
                
                # Check if independent responses are enabled
                if hasattr(self, 'independent_ai_checkbox') and self.independent_ai_checkbox.isChecked():
                    logging.info(f"🔄 Generating independent AI response (no conversation context)")
                    self.text_gen_manager.generate_independent_response_streaming(user_input, streaming_callback)
                else:
                    logging.info(f"🔄 Generating AI response with conversation context")
                    self.text_gen_manager.generate_response_streaming(user_input, streaming_callback)
                
            except Exception as e:
                logging.error(f"Error generating AI response: {e}")
                self.ai_response_signal.emit(f"Error: {str(e)}", True)
        
        # Run in background thread
        threading.Thread(target=generate_worker, daemon=True).start()
    
    def handle_ai_response(self, response_chunk: str, is_complete: bool):
        """Handle AI response chunks (for streaming)"""
        try:
            if is_complete:
                # Response is complete
                self.ai_status_label.setText("Status: AI response generated successfully")
                # Add timestamp to response
                current_time = time.strftime("%H:%M:%S")
                current_text = self.ai_response_text.toPlainText()
                if current_text:
                    separator = f"\n\n--- [{current_time}] ---\n"
                    self.ai_response_text.append(separator)
                # Auto-scroll to bottom
                cursor = self.ai_response_text.textCursor()
                cursor.movePosition(cursor.End)
                self.ai_response_text.setTextCursor(cursor)
                # Play AI response as speech through virtual mic (Groq TTS)
                if self.pulse_manager.virtual_source_exists():
                    # Extract only the latest response text (not accumulated)
                    response_text = self.current_ai_response.strip() if hasattr(self, 'current_ai_response') else response_chunk.strip()
                    if response_text:
                        logging.info("🔊 Sending AI response to Groq TTS for virtual mic playback...")
                        # Play only the latest response, then reset buffer
                        threading.Thread(target=self.tts_manager.speak_groq, args=(response_text,), daemon=True).start()
                    self.last_ai_response_text = response_text  # Track last spoken response
                    self.current_ai_response = ""  # Reset after playback
            else:
                # Streaming chunk
                if response_chunk:
                    # Start new response if this is the first chunk
                    if not self.current_ai_response:
                        current_time = time.strftime("%H:%M:%S")
                        current_text = self.ai_response_text.toPlainText()
                        if current_text:
                            separator = f"\n\n--- [{current_time}] AI Response ---\n"
                            self.ai_response_text.append(separator)
                        else:
                            header = f"--- [{current_time}] AI Response ---\n"
                            self.ai_response_text.insertPlainText(header)
                    # Append streaming chunk
                    self.ai_response_text.insertPlainText(response_chunk)
                    self.current_ai_response += response_chunk
                    # Auto-scroll to bottom
                    cursor = self.ai_response_text.textCursor()
                    cursor.movePosition(cursor.End)
                    self.ai_response_text.setTextCursor(cursor)
        except Exception as e:
            logging.error(f"Error handling AI response: {e}")
    
    def clear_ai_response(self):
        """Clear AI response text"""
        self.ai_response_text.clear()
        self.current_ai_response = ""
        self.ai_status_label.setText("Status: Ready for AI response generation")
    
    def copy_ai_to_tts(self):
        """Copy AI response text to TTS input"""
        ai_response = self.ai_response_text.toPlainText().strip()
        if ai_response:
            # Extract just the latest response (after the last timestamp)
            lines = ai_response.split('\n')
            response_text = ""
            collecting = False
            
            for line in lines:
                if '--- [' in line and 'AI Response' in line:
                    collecting = True
                    response_text = ""
                elif collecting and not line.startswith('---'):
                    response_text += line + '\n'
            
            if response_text.strip():
                self.tts_text.setPlainText(response_text.strip())
                self.status_bar.showMessage("AI response copied to TTS input")
            else:
                self.tts_text.setPlainText(ai_response)
                self.status_bar.showMessage("AI response copied to TTS input")
        else:
            QMessageBox.information(self, "No Text", "No AI response text to copy.")
    
    def clear_ai_conversation(self):
        """Clear AI conversation history"""
        self.text_gen_manager.clear_conversation()
        self.ai_status_label.setText("Status: AI conversation history cleared")
        self.status_bar.showMessage("AI conversation history cleared")
    
    def is_ai_own_speech(self, transcription: str) -> bool:
        """Check if the transcription is likely the AI's own speech to prevent feedback loops"""
        if not self.last_ai_response_text:
            return False
        
        transcription_clean = transcription.lower().strip()
        ai_response_clean = self.last_ai_response_text.lower().strip()
        
        # Check for significant overlap between transcription and last AI response
        # This helps detect when the system is transcribing its own speech
        if len(transcription_clean) > 10 and len(ai_response_clean) > 10:
            # Simple similarity check - if more than 60% of words match, likely AI speech
            trans_words = set(transcription_clean.split())
            ai_words = set(ai_response_clean.split())
            
            if trans_words and ai_words:
                overlap = len(trans_words.intersection(ai_words))
                similarity = overlap / min(len(trans_words), len(ai_words))
                
                if similarity > 0.6:  # 60% word overlap threshold
                    return True
        
        # Check for exact substring matches (common phrases from AI)
        ai_phrases = [
            "i understand", "i can help", "let me", "i think", "i believe",
            "according to", "in my opinion", "i would suggest", "it seems",
            "i apologize", "sorry", "i'm sorry", "certainly", "of course"
        ]
        
        for phrase in ai_phrases:
            if phrase in transcription_clean and phrase in ai_response_clean:
                return True
        
        return False
    
    def refresh_api_keys(self):
        """Refresh API keys from environment variables"""
        try:
            self.status_bar.showMessage("Refreshing API keys...")
            
            # Reload environment variables
            from dotenv import load_dotenv
            load_dotenv(override=True)  # Override existing values
            
            # Reinitialize Groq clients
            old_groq_available = bool(self.text_gen_manager.groq_client and self.stt_manager.groq_client)
            
            # Reinitialize text generation manager
            self.text_gen_manager.init_groq_client()
            
            # Reinitialize STT manager  
            self.stt_manager.init_groq_client()
            
            # Reinitialize TTS manager Groq client
            api_key = os.getenv('GROQ_API_KEY')
            if api_key and GROQ_AVAILABLE:
                self.tts_manager.groq_client = Groq(api_key=api_key)
                self.tts_manager.groq_tts_manager = GroqTTSManager(self.tts_manager.groq_client, self.pulse_manager.virtual_sink_name)
            
            new_groq_available = bool(self.text_gen_manager.groq_client and self.stt_manager.groq_client)
            
            if new_groq_available:
                self.ai_status_label.setText("Status: API keys refreshed successfully")
                self.status_bar.showMessage("API keys refreshed successfully")
                QMessageBox.information(
                    self, "API Keys Refreshed",
                    "Groq API keys have been successfully refreshed from environment variables.\n\n"
                    "You should now be able to use the updated API key."
                )
            else:
                self.ai_status_label.setText("Status: API key refresh failed")
                self.status_bar.showMessage("API key refresh failed")
                QMessageBox.warning(
                    self, "API Key Refresh Failed",
                    "Failed to refresh API keys. Please check:\n\n"
                    "1. GROQ_API_KEY is set in your environment\n"
                    "2. The API key is valid\n"
                    "3. Check the application log for details"
                )
                
        except Exception as e:
            logging.error(f"Error refreshing API keys: {e}")
            self.ai_status_label.setText("Status: API key refresh error")
            self.status_bar.showMessage("API key refresh error")
            QMessageBox.critical(
                self, "Error",
                f"An error occurred while refreshing API keys:\n\n{str(e)}"
            )
    

    

    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop real-time transcription if active
        if self.is_realtime_transcription_active:
            self.stt_manager.stop_realtime_transcription()
        
        if self.pulse_manager.virtual_source_exists():
            reply = QMessageBox.question(
                self, "BEET",
                "Virtual microphone is still active. Remove it before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.pulse_manager.remove_virtual_microphone()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    



def check_dependencies():
    """Check if required system dependencies are available"""
    try:
        # Check if pactl is available
        result = subprocess.run(['pactl', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False, "pactl (PulseAudio) not found"
        
        # Check if PulseAudio is running
        result = subprocess.run(['pulseaudio', '--check'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False, "PulseAudio is not running"
        
        return True, "All dependencies satisfied"
        
    except FileNotFoundError:
        return False, "PulseAudio tools not found in PATH"


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("BEET")
    app.setApplicationVersion("1.0.0")
    
    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        QMessageBox.critical(
            None, "Dependency Error",
            f"Missing required dependencies:\n{deps_msg}\n\n"
            "Please install PulseAudio and ensure it's running."
        )
        return 1
    
    # Create and show main window
    window = BEETMainWindow()
    window.show()
    
    # Initial device refresh
    window.refresh_devices()
    
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main()) 