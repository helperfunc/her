"""Simplified TTS Manager that ensures female voice from first use."""
import os
import time
import tempfile
import numpy as np
from typing import Optional, Tuple
import pyttsx3
from scipy.io import wavfile
from config import config
from utils import TextProcessor

class SimpleTTSManager:
    """Simplified TTS Manager with guaranteed female voice."""
    
    def __init__(self):
        """Initialize with female voice."""
        import pyttsx3
        self.engine = pyttsx3.init()
        
        # Configure engine
        self.engine.setProperty('rate', 180)  # Natural female speech rate
        self.engine.setProperty('volume', 1.0)
        
        # Get all voices and find Zira (female English voice)
        voices = self.engine.getProperty('voices')
        self.female_voice_id = None
        
        print("\n[TTS] Searching for female voice...")
        
        # First pass: Look for Zira specifically
        for voice in voices:
            if 'zira' in voice.name.lower():
                self.female_voice_id = voice.id
                print(f"[TTS] Found preferred female voice: {voice.name}")
                break
        
        # Second pass: Look for any English female voice
        if not self.female_voice_id:
            for voice in voices:
                name = voice.name.lower()
                is_english = 'english' in name or 'en-' in str(getattr(voice, 'languages', '')).lower()
                is_female = any(fem in name for fem in ['female', 'woman', 'zira', 'eva', 'hazel'])
                
                if is_english and is_female:
                    self.female_voice_id = voice.id
                    print(f"[TTS] Found alternative female voice: {voice.name}")
                    break
        
        # Third pass: Any female voice
        if not self.female_voice_id:
            for voice in voices:
                if getattr(voice, 'gender', '').lower() == 'female':
                    self.female_voice_id = voice.id
                    print(f"[TTS] Found generic female voice: {voice.name}")
                    break
        
        # Set the female voice
        if self.female_voice_id:
            self.engine.setProperty('voice', self.female_voice_id)
            print(f"[TTS] Female voice configured for 'her' project")
        else:
            print("[WARN] No female voice found, using default")
        
        self._synthesis_count = 0
    
    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize text with female voice."""
        try:
            # Extract response text
            tts_text = TextProcessor.clean_text_for_tts(text)
            if not tts_text:
                return None
            
            print(f"\n[TTS] Synthesizing: {tts_text[:50]}...")
            
            # ALWAYS ensure female voice is set before synthesis
            if self.female_voice_id:
                self.engine.setProperty('voice', self.female_voice_id)
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_filename = fp.name
            
            # Generate speech
            self.engine.save_to_file(tts_text, temp_filename)
            self.engine.runAndWait()
            
            # Read audio
            sample_rate, wav_data = wavfile.read(temp_filename)
            os.unlink(temp_filename)
            
            # Convert to mono if stereo
            if len(wav_data.shape) > 1:
                wav_data = np.mean(wav_data, axis=1)
            
            # Normalize
            wav_data = wav_data / (np.max(np.abs(wav_data)) + 1e-10)
            wav_data = np.array(wav_data * 32767, dtype=np.int16)
            
            self._synthesis_count += 1
            print(f"[TTS] Synthesis #{self._synthesis_count} complete")
            
            return wav_data, sample_rate
            
        except Exception as e:
            print(f"[ERROR] TTS failed: {str(e)}")
            return None

# Test the simplified manager
if __name__ == "__main__":
    import sounddevice as sd
    
    print("\nTesting Simplified Female Voice TTS")
    print("="*50)
    
    tts = SimpleTTSManager()
    
    # Test multiple syntheses
    tests = [
        "This is the first test. It should be a female voice.",
        "This is the second test. Still female voice.",
        "Third test here. The voice remains female."
    ]
    
    for i, text in enumerate(tests, 1):
        print(f"\nTest {i}: {text}")
        full_text = f"<think>Test</think>\n<response>{text}</response>"
        
        result = tts.synthesize(full_text)
        if result:
            data, fs = result
            print(f"Playing test {i}...")
            sd.play(data, fs)
            sd.wait()
            time.sleep(0.5)
    
    print("\nAll tests should have used female voice!")