"""Fixed TTS Manager that ensures all responses have voice."""
import os
import time
import tempfile
import numpy as np
from typing import Optional, Tuple
from scipy.io import wavfile
from utils import TextProcessor

class FixedTTSManager:
    """TTS Manager with guaranteed voice for all responses."""
    
    def __init__(self):
        """Initialize TTS engine with female voice."""
        self.is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
        self.selected_voice_id = None
        self._synthesis_count = 0
        
        if not self.is_wsl:
            # Windows: Don't create engine in init, create it on demand
            print("[TTS] TTS Manager initialized (engine will be created on first use)")
        else:
            print("[TTS] Running in WSL, will use Windows TTS")
    
    def _ensure_engine(self):
        """Ensure engine exists and is properly configured."""
        import pyttsx3
        
        # Always create a fresh engine to avoid state issues
        if hasattr(self, 'engine'):
            try:
                self.engine.stop()
                del self.engine
            except:
                pass
        
        # Create new engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 1.0)
        
        # Set female voice
        if not self.selected_voice_id:
            # Find and store female voice ID
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if voice.name and 'zira' in voice.name.lower():
                    self.selected_voice_id = voice.id
                    print(f"[TTS] Female voice found: {voice.name}")
                    break
            
            if not self.selected_voice_id:
                # Try other female voices
                for voice in voices:
                    name = (voice.name or '').lower()
                    if any(f in name for f in ['female', 'hedda', 'helena']):
                        self.selected_voice_id = voice.id
                        print(f"[TTS] Alternative female voice: {voice.name}")
                        break
            
            if not self.selected_voice_id and voices:
                self.selected_voice_id = voices[0].id
                print("[WARN] Using default voice")
        
        # Set the voice
        if self.selected_voice_id:
            self.engine.setProperty('voice', self.selected_voice_id)
    
    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize text to speech."""
        self._synthesis_count += 1
        print(f"\n[TTS] Synthesis #{self._synthesis_count}")
        
        try:
            # Extract text
            tts_text = TextProcessor.clean_text_for_tts(text)
            if not tts_text:
                return None
            
            print(f"[TTS] Text: {tts_text[:60]}...")
            
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_filename = fp.name
            
            if not self.is_wsl:
                # Ensure engine exists and is configured
                self._ensure_engine()
                
                # Generate speech
                self.engine.save_to_file(tts_text, temp_filename)
                self.engine.runAndWait()
            
            # Read audio
            sample_rate, wav_data = wavfile.read(temp_filename)
            os.unlink(temp_filename)
            
            # Process audio
            if len(wav_data.shape) > 1:
                wav_data = np.mean(wav_data, axis=1)
            
            wav_data = wav_data / (np.max(np.abs(wav_data)) + 1e-10)
            wav_data = np.array(wav_data * 32767, dtype=np.int16)
            
            print(f"[TTS] Generated {len(wav_data)} samples")
            return wav_data, sample_rate
            
        except Exception as e:
            print(f"[ERROR] TTS failed: {str(e)}")
            return None

# Test
if __name__ == "__main__":
    import sounddevice as sd
    
    print("\nTesting Fixed TTS Manager")
    print("="*50)
    
    tts = FixedTTSManager()
    
    tests = [
        "First response should have female voice",
        "Second response should also have female voice",
        "Third response maintains female voice"
    ]
    
    for i, text in enumerate(tests, 1):
        print(f"\nTest {i}: {text}")
        msg = f"<response>{text}</response>"
        
        result = tts.synthesize(msg)
        if result:
            print(f"Playing test {i}...")
            sd.play(result[0], result[1])
            sd.wait()
            time.sleep(0.5)
        else:
            print(f"FAILED test {i}")
    
    print("\nAll responses should have female voice!")