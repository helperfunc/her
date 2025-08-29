"""Diagnose TTS issues."""
import sys
import time

print("Starting TTS diagnosis...")
print("-" * 50)

# Step 1: Import pyttsx3
print("Step 1: Importing pyttsx3...")
start = time.time()
try:
    import pyttsx3
    print(f"  OK - Import took {time.time() - start:.2f} seconds")
except Exception as e:
    print(f"  FAIL - {e}")
    sys.exit(1)

# Step 2: Initialize engine
print("\nStep 2: Initializing TTS engine...")
start = time.time()
try:
    engine = pyttsx3.init()
    print(f"  OK - Init took {time.time() - start:.2f} seconds")
except Exception as e:
    print(f"  FAIL - {e}")
    sys.exit(1)

# Step 3: Get voices
print("\nStep 3: Getting voices...")
start = time.time()
try:
    voices = engine.getProperty('voices')
    print(f"  OK - Found {len(voices)} voices in {time.time() - start:.2f} seconds")
    
    # List voices
    for i, voice in enumerate(voices):
        print(f"    Voice {i}: {voice.name}")
        if 'zira' in voice.name.lower():
            print(f"      ^ ZIRA FOUND! ID: {voice.id}")
except Exception as e:
    print(f"  FAIL - {e}")
    sys.exit(1)

# Step 4: Set female voice
print("\nStep 4: Setting female voice...")
start = time.time()
try:
    for voice in voices:
        if 'zira' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            print(f"  OK - Set to Zira in {time.time() - start:.2f} seconds")
            break
except Exception as e:
    print(f"  FAIL - {e}")

# Step 5: Test synthesis
print("\nStep 5: Testing synthesis...")
start = time.time()
try:
    engine.say("Hello, I am Her, your AI assistant.")
    engine.runAndWait()
    print(f"  OK - Synthesis took {time.time() - start:.2f} seconds")
except Exception as e:
    print(f"  FAIL - {e}")

print("\n" + "=" * 50)
print("Diagnosis complete!")