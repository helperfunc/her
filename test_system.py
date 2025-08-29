"""Test script to verify the voice system fixes."""
import sys
import os
import asyncio
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import LLMManager, TTSManager, ConversationState
from config import config

def test_llm_response():
    """Test LLM response generation with new prompt."""
    print("Testing LLM response generation...")
    
    llm = LLMManager()
    state = ConversationState()
    
    # Test messages
    test_questions = [
        "What is 2+2?",
        "Tell me about Python",
        "How are you today?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {question}")
        print('='*50)
        
        state.add_message("user", question)
        response = llm.generate_response(state.history)
        
        # Extract think and response parts
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        
        if think_match:
            think = think_match.group(1).strip()
            print(f"\nTHOUGHT: {think}")
        
        if response_match:
            reply = response_match.group(1).strip()
            print(f"\nRESPONSE: {reply}")
            print(f"Response length: {len(reply.split())} words")
        
        # Add response to history for next test
        state.add_message("assistant", response)
    
    print("\n✅ LLM response test completed")

def test_tts_synthesis():
    """Test TTS synthesis with multiple calls."""
    print("\nTesting TTS synthesis...")
    
    tts = TTSManager()
    
    test_texts = [
        """<think>Testing first response</think>
<response>This is the first test response.</response>""",
        """<think>Testing second response</think>
<response>This is the second test response.</response>""",
        """<think>Testing third response</think>
<response>This is the third test response.</response>"""
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTTS Test {i}...")
        result = tts.synthesize(text)
        
        if result and isinstance(result, tuple):
            data, fs = result
            print(f"✅ Synthesis {i} successful: {len(data)} samples at {fs}Hz")
        else:
            print(f"❌ Synthesis {i} failed")
    
    print("\n✅ TTS synthesis test completed")

def main():
    """Run all tests."""
    print("Starting voice system tests...\n")
    
    try:
        # Test LLM response generation
        test_llm_response()
        
        # Test TTS synthesis
        test_tts_synthesis()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()