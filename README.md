# Google Meet Automation & BEET Virtual Microphone Suite

This project combines Google Meet automation (with Puppeteer) and a powerful virtual microphone system (BEET) for advanced audio routing, Text-to-Speech (TTS), and Speech-to-Text (STT) on Linux.

## Features

- **Automated Google Meet Bot**
  - Logs into Google account
  - Navigates to a Google Meet link
  - Clicks the "Join now" button automatically
  - Uses stealth mode to avoid bot detection
- **BEET Virtual Microphone GUI**
  - Creates a PulseAudio virtual microphone
  - Real-time Speech-to-Text (STT) using Groq's Whisper API
  - Text-to-Speech (TTS): offline (pyttsx3) and online (gTTS, Groq)
  - System audio capture and routing
  - Audio device management and volume control
  - Modern PyQt5 GUI with real-time logging
- **Unified Launch**
  - Running the automation script launches both the BEET GUI (in a new konsole window) and the Google Meet bot in parallel

## Requirements

- Linux with PulseAudio
- Python 3.7+
- Node.js 14+
- `konsole` terminal emulator (for automatic BEET GUI launch)
- Google account for Meet automation
- (Optional) Groq API key for advanced STT/TTS features

## Setup

### 1. Clone the Repository
```sh
git clone <repository-url>
cd ownbrowser/gmeet-automation
```

### 2. Install Node.js Dependencies
```sh
npm install
```

### 3. Install Python Dependencies for BEET
```sh
cd BEET
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Credentials
- Edit `join_gmeet.js` and set:
  - `EMAIL` to your Google email
  - `PASSWORD` to your Google password
  - `MEET_URL` to your Google Meet link (e.g., `https://meet.google.com/abc-defg-hij`)
- (Optional) Set up your Groq API key in `BEET/.env` for STT/TTS features

## Usage

### Unified Launch (Recommended)
From the `gmeet-automation` directory, run:
```sh
npm start
```
_or_
```sh
node join_gmeet.js
```
- This will:
  - Open the BEET Virtual Microphone GUI in a new `konsole` window
  - Start the Google Meet automation bot in your current terminal

### Manual Launch (Advanced)
You can also run each component separately:
- **BEET GUI:**
  ```sh
  cd BEET
  python main.py
  ```
- **Google Meet Automation:**
  ```sh
  cd ..
  node join_gmeet.js
  ```

## Notes & Troubleshooting
- **konsole required:** The automation script uses `konsole` to launch the BEET GUI. If you use a different terminal emulator, edit `join_gmeet.js` and change `'konsole'` to your preferred terminal (e.g., `'xterm'`, `'gnome-terminal'`).
- **PulseAudio:** Ensure PulseAudio is running for virtual microphone features.
- **Audio Devices:** Set "BEET Virtual Microphone" as your input device in Google Meet for TTS/STT features.
- **Credentials:** For security, consider using environment variables or a secrets manager for your Google credentials in production.
- **Linux Only:** This suite is designed for Linux. macOS/Windows support is experimental.

## File Structure
```
gmeet-automation/
├── BEET/                # Virtual microphone and audio GUI (Python)
│   ├── main.py
│   ├── requirements.txt
│   └── ...
├── join_gmeet.js        # Google Meet automation (Node.js)
├── package.json
└── ...
```

## License
[Add your license information here]

## Support
- For BEET audio issues, see `BEET/README.md` troubleshooting section
- For automation issues, open an issue or discussion on the repository

---
**This project is for educational and research purposes. Automating login and meeting join may violate Google Meet's terms of service. Use responsibly.** 