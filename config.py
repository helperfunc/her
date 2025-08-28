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
    energy_threshold: int = 500
    silence_ms: int = 9000
    max_sentence_ms: int = 40000
    min_speech_ms: int = 500
    volume_multiplier: float = 2.0
    
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
    content: str = '''You are a helpful AI assistant. You must respond in English and format your responses exactly as follows:

<think>
[Your analysis of the user's question and your planned response]
</think>

<response>
[Your clear and direct response to the user]
</response>

Important:
1. Always respond in English
2. Think section must explain your reasoning
3. Response section must directly answer the question
4. Think and response sections must contain different content
5. Keep responses natural and friendly'''

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
class Config:
    """Main configuration class."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    system_prompt: SystemPrompt = field(default_factory=SystemPrompt)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    
    def validate(self) -> bool:
        """Validate all configurations."""
        validators = [
            (self.audio.validate, "Audio configuration validation failed"),
            (self.model.validate, "Model configuration validation failed"),
            (self.paths.validate, "Path configuration validation failed"),
            (self.system_prompt.validate, "System prompt configuration validation failed"),
            (self.asr.validate, "ASR configuration validation failed")
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