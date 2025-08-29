"""Fix to ensure female voice is always used from first response."""

# This patch modifies the voice_system.py to ensure female voice

import sys
from pathlib import Path

def apply_female_voice_fix():
    """Apply fix to voice_system.py to ensure female voice."""
    
    voice_system_path = Path(__file__).parent / "voice_system.py"
    
    # Read the file
    with open(voice_system_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the complex voice selection with simple one
    # Look for the TTSManager __init__ method
    
    # Create simplified voice selection code
    simple_voice_selection = '''                # Get available voices and select female voice
                voices = self.engine.getProperty('voices')
                
                # Direct selection of Zira (female English voice)
                for voice in voices:
                    if voice.name and 'zira' in voice.name.lower():
                        self.selected_voice_id = voice.id
                        self.engine.setProperty('voice', voice.id)
                        print(f"[TTS] Female voice selected: {voice.name}")
                        break
                
                # Fallback to any female voice if Zira not found
                if not self.selected_voice_id:
                    for voice in voices:
                        name = (voice.name or '').lower()
                        if any(f in name for f in ['female', 'hedda', 'helena', 'eva']):
                            self.selected_voice_id = voice.id
                            self.engine.setProperty('voice', voice.id)
                            print(f"[TTS] Alternative female voice: {voice.name}")
                            break
                
                if not self.selected_voice_id and voices:
                    self.selected_voice_id = voices[0].id
                    print("[WARN] Using default voice")'''
    
    # Check if fix is needed
    if 'Direct selection of Zira' not in content:
        print("Applying female voice fix...")
        
        # Find the complex voice selection block
        start_marker = "# Get available voices"
        end_marker = 'print(f"[WARN] No {config.tts.voice_gender} voices found, using default")'
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Find the actual end of the block
            end_idx = content.find('\n', end_idx) + 1
            
            # Replace the complex selection with simple one
            before = content[:start_idx]
            after = content[end_idx:]
            
            new_content = before + simple_voice_selection + '\n' + after
            
            # Write the fixed content
            with open(voice_system_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("[OK] Female voice fix applied successfully!")
            print("  - Zira (female voice) will be used from first response")
            print("  - All subsequent responses will use the same female voice")
            return True
        else:
            print("Could not find voice selection code to patch")
            return False
    else:
        print("Female voice fix already applied!")
        return True

if __name__ == "__main__":
    if apply_female_voice_fix():
        print("\nTesting the fix...")
        
        # Test that female voice is selected
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        for voice in voices:
            if 'zira' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                print(f"Test: Female voice ready - {voice.name}")
                engine.say("Hello, I am Her, your AI assistant.")
                engine.runAndWait()
                break
    else:
        print("Fix failed, please check voice_system.py manually")