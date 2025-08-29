"""Complete test of voice system - all responses should have female voice."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
import sounddevice as sd
import time

def test_all_voices():
    """Test that ALL responses have female voice."""
    print("\n" + "="*70)
    print("COMPLETE VOICE SYSTEM TEST")
    print("="*70)
    
    # Initialize TTS
    print("\nInitializing TTS Manager...")
    tts = TTSManager()
    
    # Test cases
    test_cases = [
        ("FIRST", "Hello, I am Her. This is my first response."),
        ("SECOND", "This is the critical second response that was failing."),
        ("THIRD", "Third response to verify consistency."),
        ("FOURTH", "Fourth response, voice should still be female."),
        ("FIFTH", "Fifth and final test response.")
    ]
    
    results = []
    
    for i, (label, text) in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {label} Response:")
        print("-" * 50)
        print(f"Text: {text}")
        
        msg = f"<think>Test</think>\n<response>{text}</response>"
        
        print("Synthesizing...")
        result = tts.synthesize(msg)
        
        if result and len(result[0]) > 0:
            print(f"[OK] Synthesis successful ({len(result[0])} samples)")
            print("Playing...")
            sd.play(result[0], result[1])
            sd.wait()
            results.append(True)
        else:
            print(f"[FAIL] No voice output!")
            results.append(False)
        
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY:")
    print("-" * 70)
    
    for i, (label, _) in enumerate(test_cases):
        status = "[OK]" if results[i] else "[FAIL]"
        print(f"  {label:10s} Response: {status}")
    
    success_rate = sum(results) / len(results) * 100
    print("-" * 70)
    print(f"Success Rate: {success_rate:.0f}% ({sum(results)}/{len(results)})")
    
    if all(results):
        print("\n[SUCCESS] All responses have female voice!")
        print("The 'HER' voice system is fully functional!")
    else:
        failed = [test_cases[i][0] for i, r in enumerate(results) if not r]
        print(f"\n[ERROR] Failed responses: {', '.join(failed)}")
    
    print("="*70)
    
    return all(results)

if __name__ == "__main__":
    success = test_all_voices()
    sys.exit(0 if success else 1)