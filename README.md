# HER - Voice Interactive AI Assistant

A sophisticated voice-interactive AI assistant inspired by the movie "Her", featuring real-time speech recognition, natural language processing, and text-to-speech capabilities with a warm, engaging personality.

## Features

- **Real-time Voice Interaction**: Seamless speech recognition and response generation
- **Natural Conversations**: Engaging, thoughtful responses with personality
- **Female Voice Synthesis**: Configured with pleasant female voice output
- **Conversation Memory**: Maintains context across interactions
- **Cross-Platform Support**: Works on Windows, macOS, and Linux (including WSL)
- **Thought Process Display**: Shows AI's reasoning alongside responses

## Project Structure

```
her/
├── main.py              # Main entry point
├── voice_system.py      # Core voice system implementation
├── config.py            # Configuration settings
├── tasks.py             # Task processing modules
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── models/              # Model files directory
│   └── llama/          # LLM model directory
│       └── Qwen3-1.7B-Q6_K.gguf
└── output/             # Output files directory
```

## Technology Stack

- **Speech Recognition**: OpenAI Whisper (tiny model for fast processing)
- **Language Model**: Llama.cpp with Qwen3-1.7B model
- **Text-to-Speech**: pyttsx3 (Windows) / PowerShell (WSL)
- **Audio Processing**: sounddevice, numpy, scipy
- **Async Processing**: asyncio for concurrent task handling

## Installation

### Prerequisites

- Python 3.8 or higher
- Working microphone and speakers
- At least 4GB RAM (8GB recommended)
- ~2GB disk space for models

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/her.git
cd her
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the language model:**
   - Download the Qwen3-1.7B-Q6_K.gguf model
   - Place it in `models/llama/` directory
   - Or modify `config.py` to point to your model location

4. **Configure audio devices (optional):**
   - The system will auto-detect audio devices
   - Check console output for selected devices
   - Modify device selection in `voice_system.py` if needed

## Usage

### Basic Usage

Run the main application:
```bash
python main.py
```

The system will:
1. Initialize all components
2. Test audio output with a tone
3. Start listening for your voice
4. Display "Ready to listen..." when ready

### Interaction Flow

1. **Speak** when you see "Ready to listen..."
2. **Wait** for the system to process (shows "LLM thinking...")
3. **Listen** to Her's response
4. **Continue** the conversation naturally

### Commands

- Say "goodbye", "bye", "exit", or "quit" to end the conversation
- The system saves conversation logs automatically

### Testing Components

Test voice synthesis:
```bash
python test_voice.py
```

Test response formatting:
```bash
python test_response_format.py
```

## Configuration

Key settings in `config.py`:

### Audio Settings
- `energy_threshold`: Voice detection sensitivity (default: 400)
- `silence_ms`: Silence duration before processing (default: 15000ms)
- `max_sentence_ms`: Maximum recording length (default: 60000ms)

### Model Settings
- `max_tokens`: Maximum response length (default: 512)
- `temperature`: Response creativity (0.0-1.0, default: 0.7)

### TTS Settings
- `voice_gender`: Voice type (default: "female")
- `voice_rate`: Speech speed (default: 180 wpm)

## Customization

### Personality

Modify the system prompt in `config.py`:
```python
content: str = '''You are Her, a friendly and intelligent AI assistant...'''
```

### Voice Selection

Change preferred voices in `config.py`:
```python
preferred_voices: list = ['zira', 'eva', 'hazel', ...]
```

### Response Style

Adjust generation parameters in `voice_system.py`:
- Increase `max_tokens` for longer responses
- Modify `temperature` for creativity level
- Adjust `presence_penalty` and `frequency_penalty` for variety

## Troubleshooting

### No Audio Output
- Check audio device selection in console output
- Ensure speakers/headphones are connected
- Try running `test_voice.py` to test TTS

### Speech Not Detected
- Check microphone permissions
- Adjust `energy_threshold` in config.py
- Ensure microphone is not muted

### Slow Response Time
- Consider using a smaller language model
- Reduce `max_tokens` for shorter responses
- Check CPU usage and available RAM

### WSL Issues
- Ensure PowerShell is accessible from WSL
- Install Windows audio drivers in WSL
- Use PulseAudio for audio routing

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the movie "Her" directed by Spike Jonze
- Built with open-source AI models and libraries
- Thanks to the Whisper, Llama.cpp, and pyttsx3 communities

## Contact

For questions or support, please open an issue on GitHub. 