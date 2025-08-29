"""Test that multiple responses all have voice output."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
import sounddevice as sd
import numpy as np

def test_multiple_responses():
    """Test that all responses (including second one) have voice output."""
    print("\n" + "="*60)
    print("Testing Multiple Voice Responses")
    print("="*60)
    
    # Initialize TTS Manager
    print("\nInitializing TTS Manager...")
    tts = TTSManager()
    print(f"Selected voice ID: {tts.selected_voice_id}")
    
    # Test messages
    test_messages = [
        "<think>First test</think>\n<response>This is the first response. Should have female voice.</response>",
        "<think>Second test</think>\n<response>This is the second response. Should also have female voice.</response>",
        "<think>Third test</think>\n<response>This is the third response. Voice should still work.</response>",
        "<think>Fourth test</think>\n<response>Fourth response here. Testing voice consistency.</response>"
    ]
    
    success_count = 0
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n" + "-"*50)
        print(f"TEST {i}: Response #{i}")
        print("-"*50)
        
        # Extract response text for display
        response_text = message.split("<response>")[1].split("</response>")[0]
        print(f"Text: {response_text}")
        
        # Synthesize
        print("Synthesizing...")
        result = tts.synthesize(message)
        
        if result and isinstance(result, tuple):
            data, fs = result
            
            # Check if audio data is valid
            if len(data) > 0 and np.max(np.abs(data)) > 0:
                print(f"[OK] Synthesis successful: {len(data)} samples")
                
                # Play the audio
                print(f"Playing response {i}...")
                sd.play(data, fs)
                sd.wait()
                
                success_count += 1
                print(f"[OK] Response {i} played successfully")
            else:
                print(f"[FAIL] Response {i} has no audio data")
        else:
            print(f"[FAIL] Response {i} synthesis failed")
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Successful: {success_count}/{len(test_messages)}")
    
    if success_count == len(test_messages):
        print("[OK] All responses had voice output!")
        print("The second response issue has been FIXED!")
    else:
        print(f"[FAIL] Only {success_count} responses had voice")
        print("The issue may still exist")
    
    return success_count == len(test_messages)

if __name__ == "__main__":
    success = test_multiple_responses()
    sys.exit(0 if success else 1)