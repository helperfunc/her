"""Test specifically that the first response has voice."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
import sounddevice as sd
import numpy as np
import time

def test_first_response():
    """Test that the first response has voice output."""
    print("\n" + "="*60)
    print("Testing FIRST Response Voice Output")
    print("="*60)
    
    # Create a NEW TTS Manager (simulating first use)
    print("\nCreating fresh TTS Manager...")
    tts = TTSManager()
    print(f"Voice ID stored: {tts.selected_voice_id}")
    
    # FIRST synthesis
    print("\n[CRITICAL TEST] First Response:")
    print("-" * 40)
    
    first_msg = "<think>First thought</think>\n<response>This is the very first response. It should have female voice.</response>"
    
    print("Attempting first synthesis...")
    result = tts.synthesize(first_msg)
    
    if result and isinstance(result, tuple):
        data, fs = result
        
        # Check audio data validity
        if len(data) > 0 and np.max(np.abs(data)) > 0:
            print(f"[OK] First synthesis successful: {len(data)} samples")
            print("Playing first response (should be female voice)...")
            sd.play(data, fs)
            sd.wait()
            print("[OK] First response played successfully!")
            
            # Test second response too
            time.sleep(1)
            print("\n[VERIFICATION] Second Response:")
            print("-" * 40)
            
            second_msg = "<think>Second thought</think>\n<response>This is the second response for verification.</response>"
            result2 = tts.synthesize(second_msg)
            
            if result2:
                print(f"[OK] Second synthesis successful: {len(result2[0])} samples")
                print("Playing second response...")
                sd.play(result2[0], result2[1])
                sd.wait()
                print("[OK] Second response played successfully!")
                
                print("\n" + "="*60)
                print("SUCCESS: Both first and second responses have voice!")
                print("="*60)
                return True
            else:
                print("[FAIL] Second response has no voice")
        else:
            print(f"[FAIL] First response has no audio data (max amplitude: {np.max(np.abs(data)) if len(data) > 0 else 0})")
            return False
    else:
        print(f"[FAIL] First synthesis failed completely")
        return False
    
    return False

if __name__ == "__main__":
    success = test_first_response()
    if not success:
        print("\n[ERROR] First response voice issue still exists!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] First response voice is working!")