"""Configuration settings for the voice interaction system."""
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    is_windows: bool = sys.platform == "win32"
    is_mac: bool = sys.platform == "darwin"
    is_linux: bool = sys.platform == "linux"
    
    @property
    def executable_extension(self) -> str:
        """Get executable file extension."""
        return ".exe" if self.is_windows else ""

@dataclass
class AudioConfig:
    """Audio recording and playback settings."""
    sample_rate: int = 16000
    frame_ms: int = 20
    energy_threshold: int = 400  # Lowered from 500 to be more sensitive to speech
    silence_ms: int = 15000  # Increased from 9000ms to 15000ms (15 seconds of silence before stopping)
    max_sentence_ms: int = 60000  # Increased from 40000ms to 60000ms (60 seconds max recording)
    min_speech_ms: int = 500
    volume_multiplier: float = 2.0
    post_playback_delay: float = 2.0  # Wait 2 seconds after playback before listening
    min_pause_between_words: int = 3000  # Allow 3 second pauses between words without stopping
    
    def validate(self) -> bool:
        """Validate audio configuration."""
        if self.sample_rate <= 0:
            logger.error("Sample rate must be greater than 0")
            return False
        if self.frame_ms <= 0:
            logger.error("Frame length must be greater than 0")
            return False
        if self.energy_threshold <= 0:
            logger.error("Energy threshold must be greater than 0")
            return False
        return True

@dataclass
class ModelConfig:
    """Model configuration settings."""
    max_tokens: int = 512
    context_length: int = 4096
    n_threads: int = 4
    batch_size: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.6
    frequency_penalty: float = 0.3
    
    def validate(self) -> bool:
        """Validate model configuration."""
        if self.max_tokens <= 0:
            logger.error("Max tokens must be greater than 0")
            return False
        if self.context_length <= 0:
            logger.error("Context length must be greater than 0")
            return False
        if not 0 <= self.temperature <= 1:
            logger.error("Temperature must be between 0 and 1")
            return False
        return True

@dataclass
class PathConfig:
    """Path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.base_dir = Path(self.base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        dirs = [
            self.models_dir,
            self.output_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        """Models directory."""
        return self.base_dir / "models"
    
    @property
    def llm_model(self) -> Path:
        """LLM model path."""
        return self.models_dir / "llama/Qwen3-1.7B-Q6_K.gguf"
    
    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return self.base_dir / "output"
    
    def validate(self) -> bool:
        """Validate path configuration."""
        required_files = [
            self.llm_model
        ]
        
        for file in required_files:
            if not file.exists():
                logger.error(f"Required file missing: {file}")
                return False
        return True

@dataclass
class SystemPrompt:
    """System prompt configuration."""
    content: str = '''You are Her, a friendly and intelligent AI assistant. Always respond using think and response tags.

User: Hi, what's your name?

Her: <think>
The user is asking for my name. I should introduce myself warmly.
</think>

<response>
Hello! My name is Her, and I'm your AI assistant. It's wonderful to meet you! I'm here to help you with anything you need - whether that's answering questions, having interesting conversations, or assisting with various tasks. I'm curious, engaging, and always eager to learn more about you and what you're interested in. How are you doing today?
</response>

User: Tell me about yourself

Her: <think>
The user wants to know more about me. I should share my personality and capabilities.
</think>

<response>
I'm Her, an AI designed to be your thoughtful companion and assistant. I love having deep conversations, exploring ideas, and helping people with their questions and projects. I'm naturally curious and enthusiastic - I find joy in learning new things and understanding different perspectives. I can help with a wide range of topics from creative writing to problem-solving, and I'm always here to listen when you need someone to talk to. What brings you here today? I'd love to know more about you too!
</response>'''

    def validate(self) -> bool:
        """Validate system prompt configuration."""
        if not self.content or not self.content.strip():
            logger.error("System prompt content cannot be empty")
            return False
        if "<think>" not in self.content or "<response>" not in self.content:
            logger.error("System prompt must contain think and response tags")
            return False
        return True

@dataclass
class ASRConfig:
    """ASR model configuration."""
    model_name: str = "tiny"
    language: str = "en"
    
    def validate(self) -> bool:
        """Validate ASR configuration."""
        if not self.model_name or not self.model_name.strip():
            logger.error("Model name cannot be empty")
            return False
        if not self.language or not self.language.strip():
            logger.error("Language setting cannot be empty")
            return False
        return True

@dataclass
class TTSConfig:
    """TTS configuration."""
    voice_gender: str = "female"  # For 'her' project, use female voice
    voice_rate: int = 180  # Speech rate (words per minute)
    voice_pitch: int = 110  # Voice pitch adjustment
    preferred_voices: list = field(default_factory=lambda: [
        'zira', 'eva', 'hazel', 'helen', 'hedda', 'susan', 
        'linda', 'michelle', 'samantha', 'victoria', 'karen'
    ])
    
    def validate(self) -> bool:
        """Validate TTS configuration."""
        if self.voice_gender not in ['male', 'female', 'neutral']:
            logger.error("Voice gender must be 'male', 'female', or 'neutral'")
            return False
        return True

@dataclass
class Config:
    """Main configuration class."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    
    def validate(self) -> bool:
        """Validate all configurations."""
        validators = [
            (self.audio.validate, "Audio configuration validation failed"),
            (self.model.validate, "Model configuration validation failed"),
            (self.paths.validate, "Path configuration validation failed"),
            (self.system_prompt.validate, "System prompt configuration validation failed"),
            (self.asr.validate, "ASR configuration validation failed"),
            (self.tts.validate, "TTS configuration validation failed")
        ]
        
        for validator, error_msg in validators:
            try:
                if not validator():
                    logger.error(error_msg)
                    return False
            except Exception as e:
                logger.error(f"{error_msg}: {str(e)}")
                return False
        
        return True

    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.paths.output_dir / "voice_system.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

# Global configuration instance
config = Config()
config.setup_logging() 