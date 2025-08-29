"""Test female voice selection for 'her' project."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import TTSManager
from config import config
import time

def test_female_voice():
    """Test that female voice is properly selected."""
    print("\n" + "="*60)
    print("Testing Female Voice Selection for 'HER' Project")
    print("="*60)
    
    # Show current TTS configuration
    print(f"\nTTS Configuration:")
    print(f"  Voice Gender: {config.tts.voice_gender}")
    print(f"  Voice Rate: {config.tts.voice_rate}")
    print(f"  Preferred Voices: {', '.join(config.tts.preferred_voices[:5])}...")
    
    print("\nInitializing TTS Manager...")
    tts = TTSManager()
    
    # Test messages specifically for 'her' project
    test_messages = [
        "<think>Greeting the user</think>\n<response>Hello! I'm Her, your personal AI assistant.</response>",
        "<think>Responding to question</think>\n<response>I'm here to help you with anything you need.</response>",
        "<think>Showing personality</think>\n<response>That's a great question! Let me think about it.</response>"
    ]
    
    print("\n" + "-"*60)
    print("Testing voice synthesis with female voice...")
    print("-"*60)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        response_match = message.split("<response>")[1].split("</response>")[0]
        print(f"  Text: {response_match}")
        
        result = tts.synthesize(message)
        if result:
            data, fs = result
            print(f"  [OK] Synthesis successful: {len(data)} samples at {fs}Hz")
            
            # Play a short sample
            try:
                import sounddevice as sd
                print(f"  Playing voice sample...")
                sd.play(data, fs)
                sd.wait()
                time.sleep(0.5)
            except:
                print(f"  (Audio playback not available in test)")
        else:
            print(f"  [FAIL] Synthesis failed")
    
    print("\n" + "="*60)
    print("Female Voice Test Complete!")
    print("="*60)

if __name__ == "__main__":
    test_female_voice()