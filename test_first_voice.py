"""Test that the first voice is female."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
from config import config
import sounddevice as sd
import time

def test_first_voice():
    """Test that the very first synthesis uses female voice."""
    print("\n" + "="*60)
    print("Testing FIRST Voice is Female for 'HER' Project")
    print("="*60)
    
    # Create a fresh TTS manager
    print("\nInitializing TTS Manager...")
    tts = TTSManager()
    
    print(f"\nStored voice ID: {tts.selected_voice_id}")
    
    # Test FIRST synthesis
    print("\n" + "-"*60)
    print("TEST 1: First synthesis (should be female voice)")
    print("-"*60)
    
    first_message = "<think>First response</think>\n<response>Hello, this is the first response.</response>"
    
    result = tts.synthesize(first_message)
    if result:
        data, fs = result
        print(f"[OK] First synthesis successful")
        print(f"Playing first response (should be female voice)...")
        sd.play(data, fs)
        sd.wait()
    else:
        print(f"[FAIL] First synthesis failed")
    
    time.sleep(1)
    
    # Test SECOND synthesis
    print("\n" + "-"*60)
    print("TEST 2: Second synthesis (should also be female voice)")
    print("-"*60)
    
    second_message = "<think>Second response</think>\n<response>This is the second response.</response>"
    
    result = tts.synthesize(second_message)
    if result:
        data, fs = result
        print(f"[OK] Second synthesis successful")
        print(f"Playing second response (should still be female voice)...")
        sd.play(data, fs)
        sd.wait()
    else:
        print(f"[FAIL] Second synthesis failed")
    
    print("\n" + "="*60)
    print("Test Complete - Both voices should be female (Zira)")
    print("="*60)

if __name__ == "__main__":
    test_first_voice()