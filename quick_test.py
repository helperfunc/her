"""Quick test of TTS voice."""
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Available voices:")
for voice in voices:
    print(f"- {voice.name}")
    if hasattr(voice, 'languages'):
        print(f"  Languages: {voice.languages}")

# Find Zira
for voice in voices:
    if 'zira' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        print(f"\nSelected: {voice.name}")
        break

# Test
engine.say("Hello, I am Her, your AI assistant.")
engine.runAndWait()
print("Done!")