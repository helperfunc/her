"""Utility functions for the voice interaction system."""
import re
import logging
from typing import List, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Collection of text processing functionalities"""
    
    @staticmethod
    def clean_text_for_tts(text: str) -> str:
        """Clean text for TTS synthesis"""
        logger.debug(f"Starting text cleanup: {text}")
        
        if not text or not text.strip():
            logger.warning("Input text is empty")
            return ""
            
        # Extract response content
        response_match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if response_match:
            text = response_match.group(1).strip()
        
        # Remove any remaining tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>.*?</[^>]+>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special sections
        text = re.sub(r'Thinking:.*?Response:', '', text, flags=re.DOTALL)
        text = re.sub(r'\[Thinking\].*?\[Response\]', '', text, flags=re.DOTALL)
        
        # Clean up formatting
        text = re.sub(r'\[(.*?)\]', '', text)  # Remove square brackets
        text = re.sub(r'[*_~`#]', '', text)    # Remove markdown
        text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
        text = re.sub(r'([.,!?])\s*', r'\1 ', text)  # Fix punctuation spacing
        
        # Remove duplicate sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = []
        for sentence in sentences:
            if sentence and not any(TextProcessor.check_similarity(sentence, existing) > 0.8 
                                  for existing in unique_sentences):
                unique_sentences.append(sentence)
        
        text = '. '.join(unique_sentences)
        
        logger.debug(f"Cleaned text: {text}")
        return text.strip()
    
    @staticmethod
    def _extract_response_content(text: str) -> str:
        """Extract response content from text"""
        patterns = [
            r'<response>(.*?)</response>',
            r'\[Response\](.*?)(?:\[|$)',
            r'\[Your clear and direct response to the user\](.*?)(?:\[|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    @staticmethod
    def _clean_formatting(text: str) -> str:
        """Clean text formatting marks"""
        # Remove XML tags
        text = re.sub(r'<[^>]+>.*?</[^>]+>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove Markdown formatting
        text = re.sub(r'\[(.*?)\].*?(?=\[|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'[*_~`#]', '', text)
        
        # Clean special characters
        text = re.sub(r'[^\x20-\x7E\n.,!?]', '', text)
        
        # Normalize whitespace and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.,!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def _remove_duplicates(text: str) -> str:
        """Remove duplicate content"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = []
        
        for sentence in sentences:
            if sentence and not any(TextProcessor.check_similarity(sentence, existing) > 0.8 
                                  for existing in unique_sentences):
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)
    
    @staticmethod
    def check_similarity(text1: str, text2: str) -> float:
        """Check similarity between two text segments"""
        return SequenceMatcher(None, text1, text2).ratio()

class AudioProcessor:
    """Collection of audio processing functionalities"""
    
    @staticmethod
    def process_audio_data(data: np.ndarray, volume_multiplier: float = 2.0) -> np.ndarray:
        """Process audio data"""
        data = data.astype(np.float32)
        data = data * volume_multiplier
        return np.clip(data, -32768, 32767).astype(np.int16)
    
    @staticmethod
    def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """Detect if audio is silent"""
        return np.max(np.abs(audio_data)) < threshold

class ResponseProcessor:
    """Collection of response processing functionalities"""
    
    @staticmethod
    def extract_thought_and_reply(full_reply: str) -> Tuple[str, str]:
        """Extract thought process and reply content"""
        if not full_reply or not full_reply.strip():
            return "", ""
        
        think_match = re.search(r'<think>(.*?)</think>', full_reply, re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""
        
        reply = full_reply[full_reply.find('</think>') + 8:].strip() if think_match else full_reply
        
        logger.debug(f"Extracted thought: {thought[:100]}...")
        logger.debug(f"Extracted reply: {reply[:100]}...")
        
        return thought, reply
    
    @staticmethod
    def format_thought_content(thought: str) -> List[str]:
        """Format thought content"""
        formatted_lines = []
        for line in thought.split('\n'):
            line = line.strip()
            if line:
                if line.startswith(('- ', '* ')) or (line[0].isdigit() and '. ' in line):
                    formatted_lines.append(f"  {line}")
                else:
                    formatted_lines.append(f"  * {line}")
        return formatted_lines
    
    @staticmethod
    def validate_reply_content(reply: str) -> bool:
        """Validate reply content"""
        if not reply or not reply.strip():
            return False
        if "<think>" in reply or "</think>" in reply:
            return False
        return True
    
    @staticmethod
    def detect_content_duplication(text: str, threshold: float = 0.85) -> bool:
        """Detect if content has duplicates"""
        sentences = text.split('.')
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if TextProcessor.check_similarity(sentences[i], sentences[j]) > threshold:
                    return True
        return False 