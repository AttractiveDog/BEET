# BEET - Virtual Microphone Application 🎤

BEET is a Python-based virtual microphone application that creates a virtual audio device for capturing and processing audio with advanced features including Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.

## Features ✨

- **Virtual Microphone Creation**: Creates a virtual microphone device using PulseAudio
- **Speech-to-Text (STT)**: Real-time audio transcription using Groq's Whisper API
- **Text-to-Speech (TTS)**: 
  - Offline TTS using pyttsx3
  - Online TTS using Google Text-to-Speech (gTTS)
- **System Audio Capture**: Capture and process audio that's playing on your system
- **GUI Interface**: User-friendly PyQt5 interface with real-time controls
- **Audio Device Management**: List and manage audio input/output devices
- **Volume Control**: Adjustable audio levels and TTS rate control
- **Real-time Logging**: Live status updates and error reporting

## Screenshots 📸

The application features a modern GUI with:
- Virtual microphone controls (Create/Remove)
- TTS input with offline/online speech synthesis
- STT recording with live transcription
- Audio device listing and management
- Real-time logging and status updates

## Requirements 📋

### System Requirements
- Linux with PulseAudio support
- Python 3.7+
- Working audio system

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- PyQt5 (GUI framework)
- pulsectl (PulseAudio control)
- numpy (Audio processing)
- pyttsx3 (Offline TTS)
- gtts (Google TTS)
- groq (Speech-to-Text API)
- sounddevice (Audio recording)
- scipy (Audio file processing)
- pygame (Audio playback)
- python-dotenv (Environment variables)

## Installation 🚀

### 1. Clone the Repository
```bash
git clone <repository-url>
cd BEET
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Setup Helper
```bash
python setup_beet.py
```

This will:
- Create a `.env` file for configuration
- Guide you through Groq API key setup
- Verify system readiness

### 5. Configure Groq API (For STT Features)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Edit the `.env` file and replace `your_groq_api_key_here` with your actual API key

## Usage 🎯

### Starting the Application
```bash
python main.py
```

**Or:**

If you run the Google Meet automation script (`join_gmeet.js`) from the parent directory, the BEET GUI will be launched automatically in a new konsole window. This requires `konsole` to be installed. If you use a different terminal emulator, you can edit `join_gmeet.js` to use your preferred terminal (e.g., `xterm`, `gnome-terminal`).

### Basic Workflow

#### 1. Create Virtual Microphone
- Click **"Create Virtual Microphone"** button
- The virtual device will be available system-wide
- Other applications can now use "BEET Virtual Microphone" as an input device

#### 2. Text-to-Speech (TTS)
- Enter text in the TTS input field
- Choose between:
  - **Offline Speech**: Uses local pyttsx3 engine (faster, no internet required)
  - **Online Speech**: Uses Google TTS (better quality, requires internet)
- Adjust speech rate using the slider
- Audio will be routed through the virtual microphone

#### 3. Speech-to-Text (STT)
- Click **"Start Recording"** to begin audio capture
- Speak or play audio that you want to transcribe
- Click **"Stop & Transcribe"** to process the recording
- Transcribed text will appear in the transcription area
- Use **"Copy to TTS"** to send transcribed text to TTS input

#### 4. Managing the Virtual Microphone
- **Remove Virtual Microphone**: Safely removes the virtual device
- **Refresh Devices**: Updates the list of available audio devices
- **Volume Control**: Adjust virtual microphone volume

### Testing Audio

#### Test Microphone Input
```bash
python mic_test.py
```

Options:
- `python mic_test.py` - Test default input device
- `python mic_test.py <device_id>` - Test specific device
- `python mic_test.py test-all` - Test all input devices

#### Test System Audio Capture
```bash
python test_system_audio.py
```

This tests the ability to capture audio that's currently playing on your system.

## Troubleshooting 🔧

### Common Issues

#### Virtual Microphone Not Created
- Ensure PulseAudio is running: `pulseaudio --check`
- Check if you have permissions for audio devices
- Try restarting PulseAudio: `pulseaudio -k && pulseaudio --start`

#### No Audio Input Detected
- Run `python mic_test.py` to test microphone
- Check microphone permissions and volume levels
- Verify the correct input device is selected

#### STT Not Working
- Verify Groq API key is correctly set in `.env` file
- Check internet connection
- Ensure you have Groq API credits available

#### TTS Audio Not Playing Through Virtual Mic
- Confirm virtual microphone is created and listed in system
- Check if other applications can see "BEET Virtual Microphone"
- Verify PulseAudio virtual sink is properly configured

### Audio Permissions on Linux
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# You may need to log out and log back in
```

## File Structure 📁

```
BEET/
├── main.py              # Main application entry point
├── setup_beet.py        # Setup helper script
├── mic_test.py          # Microphone testing utility
├── test_system_audio.py # System audio testing utility
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .env                # Environment variables (created by setup)
└── venv/               # Virtual environment (if created)
```

## Technical Details 🔧

### Audio Processing
- Sample Rate: 16 kHz (optimized for speech recognition)
- Audio Format: 16-bit PCM
- Channels: Mono (1 channel)

### Virtual Microphone Technology
- Uses PulseAudio's virtual source/sink system
- Creates a loopback device for audio routing
- Compatible with any application that uses PulseAudio

### API Integration
- **Groq Whisper API**: For accurate speech-to-text transcription
- **Google TTS API**: For high-quality text-to-speech synthesis
- **pyttsx3**: For offline text-to-speech capabilities

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License 📄

[Add your license information here]

## Support 💬

For issues and questions:
1. Check the troubleshooting section above
2. Run the test utilities to diagnose problems
3. Check system logs for audio-related errors
4. Create an issue with detailed error information

---

**Note**: This application is designed for Linux systems with PulseAudio. macOS and Windows support may require additional configuration or alternative audio backends. 