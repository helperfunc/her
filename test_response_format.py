"""Test script to verify response formatting is working correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from voice_system import LLMManager
from config import config

def test_response_format():
    """Test that responses are properly formatted without meta-instructions."""
    print("\n" + "="*60)
    print("Testing Response Format (No Meta-Instructions)")
    print("="*60)
    
    # Initialize LLM Manager
    print("\nInitializing LLM Manager...")
    llm = LLMManager()
    
    # Test questions
    test_questions = [
        "Hi, how are you? What's your name?",
        "Tell me about yourself",
        "What can you help me with?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {question}")
        print("="*60)
        
        # Create conversation history
        messages = [
            {"role": "system", "content": config.system_prompt.content},
            {"role": "user", "content": question}
        ]
        
        # Generate response
        response = llm.generate_response(messages)
        
        # Check for proper formatting
        print("\n[RESPONSE ANALYSIS]")
        
        # Check for meta-instructions
        if any(phrase in response for phrase in ['[Thoughtful', '[Comprehensive', '[Your', 'NOT visible']):
            print("❌ WARNING: Response contains meta-instructions!")
        else:
            print("✓ No meta-instructions detected")
        
        # Check for proper tags
        if '<think>' in response and '</think>' in response:
            print("✓ Think tags present")
        else:
            print("❌ Think tags missing")
            
        if '<response>' in response and '</response>' in response:
            print("✓ Response tags present")
        else:
            print("❌ Response tags missing")
        
        # Extract and display content
        import re
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        
        if think_match:
            print(f"\n[THINK CONTENT]:\n{think_match.group(1).strip()}")
        
        if response_match:
            print(f"\n[RESPONSE CONTENT]:\n{response_match.group(1).strip()}")
        
        print("\n" + "-"*60)

if __name__ == "__main__":
    test_response_format()