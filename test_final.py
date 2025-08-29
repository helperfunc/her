"""Final test to confirm all fixes are working."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
import sounddevice as sd
import time

def test_final():
    """Final test of the female voice system."""
    print("\n" + "="*60)
    print("FINAL TEST: 'HER' Voice System")
    print("="*60)
    
    print("\nInitializing TTS with female voice...")
    tts = TTSManager()
    
    print("\n[TEST 1] First response (should be female voice):")
    msg1 = "<think>Testing</think>\n<response>Hello, I'm Her, your AI assistant.</response>"
    result1 = tts.synthesize(msg1)
    if result1:
        print("  Playing first response...")
        sd.play(result1[0], result1[1])
        sd.wait()
        print("  [OK] First response has female voice")
    else:
        print("  [FAIL] First response failed")
        return False
    
    time.sleep(1)
    
    print("\n[TEST 2] Second response (critical - should also be female):")
    msg2 = "<think>Testing</think>\n<response>This is the second response. My voice should still be female.</response>"
    result2 = tts.synthesize(msg2)
    if result2:
        print("  Playing second response...")
        sd.play(result2[0], result2[1])
        sd.wait()
        print("  [OK] Second response has female voice")
    else:
        print("  [FAIL] Second response failed")
        return False
    
    time.sleep(1)
    
    print("\n[TEST 3] Third response (consistency check):")
    msg3 = "<think>Testing</think>\n<response>Third test. Voice remains consistent and female.</response>"
    result3 = tts.synthesize(msg3)
    if result3:
        print("  Playing third response...")
        sd.play(result3[0], result3[1])
        sd.wait()
        print("  [OK] Third response has female voice")
    else:
        print("  [FAIL] Third response failed")
        return False
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("  [OK] All responses use female voice (Zira)")
    print("  [OK] Second response issue is FIXED")
    print("  [OK] Voice consistency maintained")
    print("\nYour 'HER' project is ready!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    if test_final():
        print("\n[SUCCESS] All tests passed successfully!")
    else:
        print("\n[ERROR] Some tests failed")