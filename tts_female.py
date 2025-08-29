"""Female voice TTS Manager for 'her' project."""
import os
import time
import tempfile
import numpy as np
from typing import Optional, Tuple
from scipy.io import wavfile
from config import config
from utils import TextProcessor

class FemaleTTSManager:
    """TTS Manager that guarantees female voice from first use."""
    
    def __init__(self):
        """Initialize TTS with female voice for 'her' project."""
        try:
            # Check if running in WSL
            self.is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
            
            # Store the selected voice ID for consistent use
            self.selected_voice_id = None
            self.engine = None
            
            if not self.is_wsl:
                # Windows: Initialize pyttsx3 with female voice
                import pyttsx3
                self.engine = pyttsx3.init()
                
                # Set initial properties
                self.engine.setProperty('rate', 180)  # Natural female speech rate
                self.engine.setProperty('volume', 1.0)
                
                # Get voices and immediately find female voice
                voices = self.engine.getProperty('voices')
                
                # Direct search for Zira (best female English voice on Windows)
                for voice in voices:
                    if voice.name and 'zira' in voice.name.lower():
                        self.selected_voice_id = voice.id
                        self.engine.setProperty('voice', voice.id)
                        print(f"[TTS] Using female voice: {voice.name}")
                        break
                
                # If Zira not found, use any female voice
                if not self.selected_voice_id:
                    female_indicators = ['female', 'hedda', 'helena', 'eva', 'hazel', 'susan']
                    for voice in voices:
                        name = (voice.name or '').lower()
                        if any(fem in name for fem in female_indicators):
                            self.selected_voice_id = voice.id
                            self.engine.setProperty('voice', voice.id)
                            print(f"[TTS] Using alternative female voice: {voice.name}")
                            break
                
                if not self.selected_voice_id:
                    print("[WARN] No female voice found, using default")
                    if voices:
                        self.selected_voice_id = voices[0].id
            else:
                # WSL: Will use PowerShell with female voice
                print("[TTS] Running in WSL, will use Windows female TTS")
                self._test_powershell_access()
            
            print("[TTS] Female voice TTS initialized for 'her' project")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize TTS: {str(e)}")
            raise
    
    def _test_powershell_access(self):
        """Test PowerShell access from WSL."""
        try:
            import subprocess
            result = subprocess.run(
                ['powershell.exe', '-Command', 'Add-Type -AssemblyName System.Speech'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"PowerShell command failed: {result.stderr}")
        except Exception as e:
            raise Exception(f"Failed to access Windows PowerShell: {str(e)}")
    
    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize text with guaranteed female voice."""
        print("\n[TTS] Synthesizing with female voice...")
        
        try:
            # Extract and clean text for TTS
            tts_text = TextProcessor.clean_text_for_tts(text)
            if not tts_text:
                print("[ERROR] No valid response text found for TTS")
                return None
            
            print(f"[TTS] Text: {tts_text[:80]}...")
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_filename = fp.name
            
            if self.is_wsl:
                # WSL: Use PowerShell with female voice
                import subprocess
                wsl_path = subprocess.check_output(['wslpath', '-w', temp_filename], text=True).strip()
                
                # PowerShell script that ensures female voice
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
                
                # Get all voices and find female voice
                $voices = $synthesizer.GetInstalledVoices()
                $femaleVoice = $null
                
                # Look for Zira first
                foreach ($voice in $voices) {{
                    if ($voice.VoiceInfo.Name -like "*Zira*") {{
                        $femaleVoice = $voice.VoiceInfo.Name
                        break
                    }}
                }}
                
                # If no Zira, find any female voice
                if (-not $femaleVoice) {{
                    foreach ($voice in $voices) {{
                        if ($voice.VoiceInfo.Gender -eq "Female") {{
                            $femaleVoice = $voice.VoiceInfo.Name
                            break
                        }}
                    }}
                }}
                
                # Set the female voice
                if ($femaleVoice) {{
                    $synthesizer.SelectVoice($femaleVoice)
                }} else {{
                    $synthesizer.SelectVoiceByHints("Female")
                }}
                
                $synthesizer.SetOutputToWaveFile("{wsl_path}")
                $synthesizer.Speak("{tts_text.replace('"', '`"')}")
                $synthesizer.Dispose()
                '''
                
                result = subprocess.run(
                    ['powershell.exe', '-Command', ps_script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    raise Exception(f"PowerShell synthesis failed: {result.stderr}")
                
                time.sleep(0.5)  # Wait for file to be written
            else:
                # Windows: Use pyttsx3 with female voice
                # CRITICAL: Always set female voice before EVERY synthesis
                if self.selected_voice_id and self.engine:
                    self.engine.setProperty('voice', self.selected_voice_id)
                
                self.engine.save_to_file(tts_text, temp_filename)
                self.engine.runAndWait()
            
            # Read the generated audio file
            sample_rate, wav_data = wavfile.read(temp_filename)
            os.unlink(temp_filename)
            
            # Convert to mono if stereo
            if len(wav_data.shape) > 1:
                wav_data = np.mean(wav_data, axis=1)
            
            # Normalize audio
            wav_data = wav_data / (np.max(np.abs(wav_data)) + 1e-10)
            wav_data = np.array(wav_data * 32767, dtype=np.int16)
            
            print("[TTS] Female voice synthesis successful")
            return wav_data, sample_rate
            
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {str(e)}")
            return None

# Test the female TTS
if __name__ == "__main__":
    import sounddevice as sd
    
    print("\n" + "="*60)
    print("Testing Female Voice TTS for 'HER' Project")
    print("="*60)
    
    tts = FemaleTTSManager()
    
    # Test messages
    test_messages = [
        "Hello, I'm Her, your personal AI assistant.",
        "This should still be a female voice.",
        "Every response uses the same female voice."
    ]
    
    for i, text in enumerate(test_messages, 1):
        print(f"\nTest {i}: {text}")
        full_text = f"<think>Test</think>\n<response>{text}</response>"
        
        result = tts.synthesize(full_text)
        if result:
            data, fs = result
            print(f"Playing test {i} with female voice...")
            sd.play(data, fs)
            sd.wait()
            time.sleep(0.5)
        else:
            print(f"Synthesis failed for test {i}")
    
    print("\n" + "="*60)
    print("All tests should have used female voice (Zira)!")
    print("="*60)