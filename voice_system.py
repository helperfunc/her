"""Main voice interaction system implementation."""
import asyncio
import os
import queue
import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import whisper
from llama_cpp import Llama
from scipy.io import wavfile
import sys
import re
from difflib import SequenceMatcher

from config import config
from utils import TextProcessor, AudioProcessor, ResponseProcessor
from tasks import mic_task, asr_task, llm_tts_task

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Holds the current state of the conversation."""
    is_playing: bool = False
    is_thinking: bool = False
    last_playback_end: float = 0
    history: List[dict] = None
    conversation_log: List[str] = None
    
    def __post_init__(self):
        """Initialize conversation history with system message."""
        if self.history is None:
            self.history = [{"role": "system", "content": config.system_prompt.content}]
        if self.conversation_log is None:
            self.conversation_log = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        # 添加到历史记录
        if self.history is None:
            self.__post_init__()
        
        # 记录到对话日志
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if role == "user":
            self.conversation_log.append(f"[{timestamp}] User: {content}")
        elif role == "assistant":
            # 提取响应内容
            response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
            if response_match:
                response = response_match.group(1).strip()
                self.conversation_log.append(f"[{timestamp}] Assistant: {response}")
        
        # Get system message
        system_message = self._get_system_message()
        
        # Build new history starting with system message
        new_history = [system_message]
        
        # Add relevant messages based on role
        if role == "user":
            self._add_user_message(new_history, content)
        else:  # assistant
            self._add_assistant_message(new_history, content)
        
        # Update history
        self.history = new_history
        
        # 检查是否结束对话
        if role == "user" and any(phrase in content.lower() for phrase in ["再见", "拜拜", "结束", "退出", "goodbye", "bye", "quit", "exit"]):
            self.save_conversation()
            print("\n对话已结束，记录已保存。")
            sys.exit(0)
    
    def _get_system_message(self) -> dict:
        """Get the system message from history."""
        return self.history[0]
    
    def _add_user_message(self, history: List[dict], content: str):
        """Add user message and last assistant message to history.
        
        Args:
            history: Current history being built
            content: New user message content
        """
        # Find and add last assistant message if exists
        for msg in reversed(self.history[1:]):
            if msg["role"] == "assistant":
                history.append(msg)
                break
        
        # Add the new user message
        history.append({"role": "user", "content": content})
    
    def _add_assistant_message(self, history: List[dict], content: str):
        """Add assistant message and its preceding user message to history.
        
        Args:
            history: Current history being built
            content: New assistant message content
        """
        # Find and add last user message if exists
        for msg in reversed(self.history[1:]):
            if msg["role"] == "user":
                history.append(msg)
                break
        
        # Add the new assistant message
        history.append({"role": "assistant", "content": content})
    
    def _log_conversation_state(self):
        """Log the current state of the conversation."""
        print("\nConversation history:")
        for msg in self.history:
            role_str = msg['role'].upper()
            if role_str == "ASSISTANT":
                self._log_assistant_message(msg['content'])
            else:
                print(f"[{role_str}]: {msg['content'][:100]}...")
    
    def _log_assistant_message(self, content: str):
        """Log assistant message with think and response parts."""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
        
        think = think_match.group(1).strip() if think_match else "No think content"
        response = response_match.group(1).strip() if response_match else "No response content"
        
        print(f"[ASSISTANT] Think: {think[:100]}...")
        print(f"[ASSISTANT] Response: {response[:100]}...")

    def save_conversation(self):
        """保存对话记录到文件。"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(self.conversation_log))
            print(f"\n对话记录已保存到: {filename}")
        except Exception as e:
            print(f"\n保存对话记录时出错: {str(e)}")

class AudioManager:
    """Manages audio recording and playback."""
    def __init__(self):
        self.stream = None
        self._setup_audio_devices()
    
    def _setup_audio_devices(self):
        """Set up audio input and output devices."""
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']}")
        
        if config.platform.is_windows:
            # Prefer a real microphone over "Stereo Mix"
            mic_keywords = ["microphone", "mic", "麦克风"]
            default_input = next(
                (i for i, d in enumerate(devices)
                 if any(k.lower() in d["name"].lower() for k in mic_keywords) and d["max_input_channels"] > 0),
                None
            )
            # If no microphone found, fall back to Realtek input
            if default_input is None:
                default_input = next(
                    (i for i, d in enumerate(devices)
                     if "Realtek" in d["name"] and d["max_input_channels"] > 0),
                    None
                )
            default_output = next(
                (i for i, d in enumerate(devices)
                 if "Realtek" in d["name"] and d["max_output_channels"] > 0),
                None
            )

            if default_input is not None and default_output is not None:
                sd.default.device = [default_input, default_output]
            else:
                # Fallback to system defaults
                sd.default.device = None
        elif config.platform.is_mac:
            # macOS specific device setup
            # 优先查找内置音频设备
            built_in_output = next((i for i, d in enumerate(devices)
                                  if "Built-in" in d["name"] and d["max_output_channels"] > 0), None)
            built_in_input = next((i for i, d in enumerate(devices)
                                 if "Built-in" in d["name"] and d["max_input_channels"] > 0), None)
            
            if built_in_input is not None and built_in_output is not None:
                sd.default.device = [built_in_input, built_in_output]
            else:
                # 如果找不到内置设备，尝试查找任何可用的 CoreAudio 设备
                default_output = next((i for i, d in enumerate(devices)
                                     if d["max_output_channels"] > 0), None)
                default_input = next((i for i, d in enumerate(devices)
                                    if d["max_input_channels"] > 0), None)
                
                if default_input is not None and default_output is not None:
                    sd.default.device = [default_input, default_output]
                else:
                    # Fallback to system defaults
                    sd.default.device = None
        else:
            # Check if running in WSL
            is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
            if is_wsl:
                # Use PulseAudio in WSL
                pulse_device = next((i for i, d in enumerate(devices) 
                                   if "pulse" in d["name"].lower()), 0)
                sd.default.device = pulse_device
            else:
                # Other Unix systems: use system defaults
                sd.default.device = None
        
        input_device = sd.query_devices(kind="input")
        output_device = sd.query_devices(kind="output")
        print(f"\nSelected input device: {input_device['name']}")
        print(f"Selected output device: {output_device['name']}")
        
        # Store device settings for future reference
        self.input_device = input_device
        self.output_device = output_device
        
    def test_audio(self):
        """Test audio output with a simple tone."""
        try:
            print("\nTesting audio output...")
            duration = 0.5  # seconds
            frequency = 440  # Hz
            
            # Get sample rate from device or use default
            try:
                sample_rate = int(self.output_device["default_samplerate"])
            except:
                sample_rate = 44100  # Default sample rate
            
            print(f"Using device: {sd.default.device}")
            print(f"Sample rate: {sample_rate} Hz")
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            test_tone = np.sin(2 * np.pi * frequency * t)
            
            # Ensure stereo output
            if len(test_tone.shape) == 1:
                test_tone = np.column_stack((test_tone, test_tone))
            
            # Scale to int16 range
            test_tone = (test_tone * 32767).astype(np.int16)
            
            print("Playing test tone...")
            sd.play(test_tone, sample_rate)
            sd.wait()
            print("Audio test completed successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Audio test failed: {str(e)}")
            print(f"Current device: {sd.default.device}")
            print(f"Output device info: {self.output_device}")
            import traceback
            print(f"Stack trace:\n{traceback.format_exc()}")
            return False

    @property
    def piper_exe(self) -> str:
        # Return different executable names based on operating system
        if sys.platform == "darwin":
            return os.path.join(self.piper_dir, "piper")
        return os.path.join(self.piper_dir, "piper.exe")

class ASRManager:
    """Manages Automatic Speech Recognition."""
    def __init__(self):
        print("[ASR] Loading model...")
        self.model = whisper.load_model("tiny")
        print("[ASR] Model loaded successfully")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        if len(audio) == 0 or np.all(audio == 0) or np.max(np.abs(audio)) < 0.01:
            return ""
        
        result = self.model.transcribe(audio, language="en")
        return result["text"].strip()

class LLMManager:
    """Manages Large Language Model interactions."""
    def __init__(self):
        print("[LLM] Loading model...")
        self.model = Llama(
            model_path=str(config.paths.llm_model),
            n_ctx=config.model.context_length,
            n_threads=config.model.n_threads,
            n_batch=config.model.batch_size,
            n_gpu_layers=0,
            use_mmap=True,
            embedding=False,
            chat_format="llama-2",
            verbose=True,
            seed=42,
            logits_all=False
        )
        print("[LLM] Model loaded successfully")
    
    def generate_response(self, messages: List[dict], max_retries: int = 3) -> str:
        """Generate response from the model."""
        for attempt in range(max_retries):
            try:
                # Build conversation history
                prompt = ""
                system_message = None
                conversation_history = []
                current_user_message = None
                
                # Extract messages
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    elif msg["role"] == "user":
                        current_user_message = msg["content"]
                        conversation_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        conversation_history.append({"role": "assistant", "content": msg["content"]})
                
                # Add system message if exists
                if system_message:
                    prompt += f"{system_message}\n\n"
                
                # Add LIMITED conversation history (only last exchange)
                if len(conversation_history) > 2:
                    # Only include the last user-assistant exchange
                    last_exchange = conversation_history[-2:]  # Last 2 messages
                    prompt += "Previous exchange:\n"
                    for msg in last_exchange:
                        if msg["role"] == "user":
                            prompt += f"User: {msg['content']}\n"
                        else:
                            # For assistant messages, only show the response part
                            response_match = re.search(r'<response>(.*?)</response>', msg['content'], re.DOTALL)
                            if response_match:
                                response = response_match.group(1).strip()
                                prompt += f"Assistant: {response}\n"
                    prompt += "\n"
                
                # Add current question
                if current_user_message:
                    prompt += f"User: {current_user_message}\n\n"
                    prompt += "Her: "  # Add space after colon
                else:
                    print("[WARN] No current user message found")
                    continue
                
                print("\n[LLM] Generating response...")
                
                completion = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=512,  # Increased to allow for more detailed responses
                    temperature=0.7,
                    top_p=0.9,
                    presence_penalty=0.6,
                    frequency_penalty=0.3,  # Reduced to allow more natural repetition in longer responses
                    stop=["User:", "\nUser:", "\n\nUser", "</response>\n\nUser"],  # Adjusted stop tokens
                    stream=True
                )
                
                full_reply = ""
                print("\n[LLM] Response:")
                print("-" * 50)
                
                try:
                    for chunk in completion:
                        if hasattr(chunk, 'text'):
                            text = chunk.text
                        else:
                            text = chunk.get('choices', [{}])[0].get('text', '')
                            
                        if text:
                            print(text, end="", flush=True)
                            full_reply += text
                            
                            # Stop if we have complete response tags
                            if '</response>' in full_reply:
                                print("\n[INFO] Complete response detected")
                                break
                            
                except Exception as e:
                    print(f"\n[WARN] Generation error: {str(e)}")
                    continue
                
                print("\n" + "-" * 50)
                
                if not full_reply.strip():
                    print("[WARN] Empty response, retrying...")
                    continue
                
                # Clean and format the response
                full_reply = self._format_response(full_reply)
                
                # Validate the response format
                if not self._validate_response_format(full_reply):
                    print("[WARN] Invalid response format, retrying...")
                    continue
                
                return full_reply
                
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        return """<think>
Unable to generate a proper response after multiple attempts.
</think>

<response>
I apologize, but I'm having trouble formulating a proper response right now. Could you please rephrase your question?
</response>"""
    
    def _format_response(self, text: str) -> str:
        """Format and clean the response text."""
        print(f"[DEBUG] Formatting response text:\n{text}")
        
        # First, try to extract content between existing tags if they exist
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        
        if think_match and response_match:
            # Tags found - extract and clean the content
            think_content = think_match.group(1).strip()
            response_content = response_match.group(1).strip()
            
            # Remove any format instructions that leaked into content
            think_content = re.sub(r'\[.*?\]', '', think_content)
            response_content = re.sub(r'\[.*?\]', '', response_content)
            think_content = re.sub(r'\(.*?NOT visible.*?\)', '', think_content, flags=re.IGNORECASE)
            response_content = re.sub(r'\(.*?actual.*?response.*?\)', '', response_content, flags=re.IGNORECASE)
            
            # Clean up any duplicated format instructions
            format_patterns = [
                r'Thoughtful analysis.*?implications',
                r'Comprehensive and detailed.*?question',
                r'Your internal reasoning.*?user',
                r'Your actual.*?response'
            ]
            for pattern in format_patterns:
                think_content = re.sub(pattern, '', think_content, flags=re.IGNORECASE | re.DOTALL)
                response_content = re.sub(pattern, '', response_content, flags=re.IGNORECASE | re.DOTALL)
            
            # Clean the contents
            think_content = self._clean_content(think_content)[:500]
            response_content = self._clean_content(response_content)
            
            # Remove any duplicated sentences in response
            response_sentences = re.split(r'(?<=[.!?])\s+', response_content)
            unique_sentences = []
            for sentence in response_sentences:
                if sentence and not any(self._check_content_similarity(sentence, existing) for existing in unique_sentences):
                    unique_sentences.append(sentence)
            response_content = ' '.join(unique_sentences)
            
            # Ensure content is meaningful
            if not think_content or think_content.lower() in ['', 'hello']:
                think_content = "Processing and analyzing your message"
            if not response_content:
                response_content = "I'm here to help. Could you please rephrase your question?"
                
        else:
            # No proper tags found - create formatted response from raw text
            text = re.sub(r'</think>\s*\[.*?\]', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = text.strip()
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) > 1:
                think_content = self._clean_content(sentences[0])[:500]
                response_content = self._clean_content(' '.join(sentences[1:]))
            else:
                think_content = "Analyzing your input"
                response_content = self._clean_content(text)
        
        # Ensure they're different
        if think_content == response_content:
            think_content = "Processing your question"
        
        # Format with proper tags
        formatted = f"""<think>
{think_content}
</think>

<response>
{response_content}
</response>"""
        
        print(f"[DEBUG] Formatted response:\n{formatted}")
        return formatted
    
    def _clean_content(self, text: str) -> str:
        """Clean text content."""
        # Remove any remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove markdown-style headers
        text = re.sub(r'#+\s+', '', text)
        # Remove multiple newlines
        text = re.sub(r'\n+', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def _check_content_similarity(self, text1: str, text2: str) -> bool:
        """Check if two texts are too similar."""
        if not text1 or not text2:
            return False
        return SequenceMatcher(None, text1, text2).ratio() > 0.7

    def _validate_response_format(self, text: str) -> bool:
        """Validate the response format."""
        # Check for required tags
        if not all(tag in text for tag in ['<think>', '</think>', '<response>', '</response>']):
            return False
            
        # Check tag order
        think_start = text.find('<think>')
        think_end = text.find('</think>')
        response_start = text.find('<response>')
        response_end = text.find('</response>')
        
        if not (0 <= think_start < think_end < response_start < response_end):
            return False
            
        # Check content between tags
        think_content = text[think_start + 7:think_end].strip()
        response_content = text[response_start + 10:response_end].strip()
        
        if not think_content or not response_content:
            return False
            
        # Check for duplicate content
        similarity = SequenceMatcher(None, think_content, response_content).ratio()
        if similarity > 0.8:  # 80% similarity threshold
            return False
            
        return True

    def _is_response_relevant(self, question: str, response: str) -> bool:
        """Check if the response is relevant to the question.
        
        Args:
            question: The user's question
            response: The generated response
            
        Returns:
            bool: True if response is relevant, False otherwise
        """
        # Extract key terms from question, excluding stop words
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                     "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                     'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                     'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                     'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                     'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                     'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                     't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                     'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                     "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                     'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                     'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
                     'tell', 'know', 'want', 'like', 'please', 'could', 'would', 'hi', 'hello'}
        
        question_terms = set(term.lower() for term in question.split() if term.lower() not in stop_words)
        
        # Extract response content
        start = response.find("<response>")
        end = response.find("</response>")
        if start != -1 and end != -1:
            response_content = response[start + len("<response>"):end].strip()
        else:
            return False
            
        # Check if response contains key terms from question
        response_terms = set(term.lower() for term in response_content.split() if term.lower() not in stop_words)
        common_terms = question_terms & response_terms
        
        # Calculate relevance score based on:
        # 1. Number of common key terms
        # 2. Ratio of common terms to question terms
        min_common_terms = min(2, max(1, len(question_terms) // 3))
        term_ratio = len(common_terms) / len(question_terms) if question_terms else 0
        
        # Response must have minimum common terms AND good term ratio
        return len(common_terms) >= min_common_terms and term_ratio >= 0.3

    def validate_tags(self, response: str) -> str:
        """Validate and fix response format
        Args:
            response: Complete LLM response
        Returns:
            str: Fixed response
        """
        if not response:
            return self.default_response()
            
        # Check if tags are complete
        has_think = "<think>" in response and "</think>" in response
        has_response = "<response>" in response and "</response>" in response
        
        if not (has_think and has_response):
            # Extract valid content and reformat
            content = re.sub(r'<[^>]+>', '', response).strip()
            return f"""<think>
Reformatting incomplete response
</think>

<response>
{content}
</response>"""

    def default_response(self) -> str:
        """Generate default formatted response."""
        return """<think>
Providing fallback response due to formatting issues
</think>

<response>
I apologize, but I need to format my response properly. Could you please repeat your question?
</response>"""

    def quality_check(self, response: str) -> bool:
        """Check response quality."""
        # Check tag format
        if not (response.count("<think>") == 1 and 
                response.count("</think>") == 1 and
                response.count("<response>") == 1 and
                response.count("</response>") == 1):
            return False
            
        # Check content length
        thought, reply = ResponseProcessor.extract_thought_and_reply(response)
        
        if len(thought.split()) < 10 or len(reply.split()) < 10:
            return False
            
        # Check for duplicate content
        if ResponseProcessor.detect_content_duplication(reply):
            return False
            
        return True

class TTSManager:
    """Manages Text-to-Speech synthesis using pyttsx3."""
    def __init__(self):
        """Initialize TTS engine."""
        try:
            # Check if running in WSL
            self.is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
            
            # Store the selected voice ID for consistent use
            self.selected_voice_id = None
            
            if not self.is_wsl:
                # Windows: Don't create engine in init to avoid hanging
                # Engine will be created on first synthesis
                self._synthesis_count = 0
                print("[TTS] TTS Manager initialized (engine created on demand)")
            else:
                # In WSL: prepare for PowerShell TTS
                print("[TTS] Running in WSL, will use Windows TTS")
                self._test_powershell_access()
            
            print("[TTS] TTS engine initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize TTS: {str(e)}")
            import traceback
            print(f"[DEBUG] Stack trace:\n{traceback.format_exc()}")
            raise

    def _test_powershell_access(self):
        """Test PowerShell access from WSL."""
        try:
            import subprocess
            result = subprocess.run(
                ['powershell.exe', '-Command', 'Add-Type -AssemblyName System.Speech'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"PowerShell command failed: {result.stderr}")
        except Exception as e:
            raise Exception(f"Failed to access Windows PowerShell: {str(e)}")

    def _extract_response_text(self, text: str) -> str:
        """Extract the response text from between response tags."""
        try:
            return TextProcessor.clean_text_for_tts(text)
        except Exception as e:
            print(f"[ERROR] Error extracting response text: {str(e)}")
            return text.strip()
    
    def synthesize(self, text: str) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize text to speech."""
        print("\n[TTS] Starting synthesis...")
        
        try:
            # Extract and clean text for TTS
            tts_text = self._extract_response_text(text)
            if not tts_text:
                print("[ERROR] No valid response text found for TTS")
                return None
                
            print(f"[TTS] Processing text: {tts_text[:100]}...")
            
            # Create temporary file for audio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
                temp_filename = fp.name
            
            if self.is_wsl:
                # Use Windows PowerShell for TTS in WSL
                import subprocess
                # Convert WSL path to Windows path
                wsl_path = subprocess.check_output(['wslpath', '-w', temp_filename], text=True).strip()
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
                # Try to select a female voice explicitly
                $voices = $synthesizer.GetInstalledVoices()
                $femaleVoice = $voices | Where-Object {{ $_.VoiceInfo.Gender -eq 'Female' }} | Select-Object -First 1
                if ($femaleVoice) {{
                    $synthesizer.SelectVoice($femaleVoice.VoiceInfo.Name)
                }} else {{
                    $synthesizer.SelectVoiceByHints('Female')
                }}
                $synthesizer.SetOutputToWaveFile("{wsl_path}")
                $synthesizer.Speak("{tts_text.replace('"', '`"')}")
                $synthesizer.Dispose()
                '''
                
                result = subprocess.run(
                    ['powershell.exe', '-Command', ps_script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"PowerShell output: {result.stdout}")
                    print(f"PowerShell error: {result.stderr}")
                    raise Exception(f"PowerShell synthesis failed: {result.stderr}")
                
                # Wait a moment for file to be written
                time.sleep(0.1)  # Reduced from 0.5 to 0.1 for faster response
            else:
                # Use pyttsx3 for Windows and other systems
                import pyttsx3
                
                # Track synthesis count
                if not hasattr(self, '_synthesis_count'):
                    self._synthesis_count = 0
                self._synthesis_count += 1
                
                # Create fresh engine for EVERY synthesis to ensure consistency
                try:
                    # Clean up old engine if exists
                    if hasattr(self, 'engine'):
                        try:
                            self.engine.stop()
                            del self.engine
                        except:
                            pass
                    
                    # Create new engine
                    self.engine = pyttsx3.init()
                    self.engine.setProperty('rate', 180)
                    self.engine.setProperty('volume', 1.0)
                    
                    # Find and set female voice
                    if not self.selected_voice_id:
                        voices = self.engine.getProperty('voices')
                        for voice in voices:
                            if voice.name and 'zira' in voice.name.lower():
                                self.selected_voice_id = voice.id
                                print(f"[TTS] Female voice found: {voice.name}")
                                break
                        
                        if not self.selected_voice_id:
                            for voice in voices:
                                name = (voice.name or '').lower()
                                if any(f in name for f in ['female', 'hedda', 'helena']):
                                    self.selected_voice_id = voice.id
                                    print(f"[TTS] Alternative female voice: {voice.name}")
                                    break
                    
                    # Set the female voice
                    if self.selected_voice_id:
                        self.engine.setProperty('voice', self.selected_voice_id)
                    
                    
                    # Generate speech to file
                    self.engine.save_to_file(tts_text, temp_filename)
                    self.engine.runAndWait()
                    
                except Exception as e:
                    print(f"[ERROR] TTS engine error: {str(e)}")
                    # Fallback: try with a new engine
                    try:
                        emergency_engine = pyttsx3.init()
                        if self.selected_voice_id:
                            emergency_engine.setProperty('voice', self.selected_voice_id)
                        emergency_engine.save_to_file(tts_text, temp_filename)
                        emergency_engine.runAndWait()
                    except:
                        raise
            
            # Read the generated audio file
            from scipy.io import wavfile
            sample_rate, wav_data = wavfile.read(temp_filename)
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            # Convert to mono if stereo
            if len(wav_data.shape) > 1:
                wav_data = np.mean(wav_data, axis=1)
            
            # Apply audio processing to reduce noise
            wav_data = self._process_audio(wav_data)
            
            print("[TTS] Audio generated successfully")
            return wav_data, sample_rate
            
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {str(e)}")
            import traceback
            print(f"[DEBUG] Stack trace:\n{traceback.format_exc()}")
            return None
    
    def _process_audio(self, wav_data: np.ndarray) -> np.ndarray:
        """Process audio to improve quality and reduce noise."""
        try:
            # Normalize audio
            wav_data = wav_data / (np.max(np.abs(wav_data)) + 1e-10)
            
            # Apply gentle noise reduction
            from scipy import signal
            
            # Design a lowpass filter
            b, a = signal.butter(4, 0.8, btype='low')
            
            # Apply the filter
            wav_data = signal.filtfilt(b, a, wav_data)
            
            # Convert to int16
            wav_data = np.array(wav_data * 32767, dtype=np.int16)
            
            return wav_data
        except Exception as e:
            print(f"[WARN] Audio processing failed, using raw audio: {str(e)}")
            return np.array(wav_data * 32767, dtype=np.int16)

class VoiceSystem:
    """Main voice interaction system."""
    def __init__(self):
        if not config.validate():
            raise RuntimeError("Configuration validation failed")
        
        self.state = ConversationState()
        self.audio_manager = AudioManager()
        self.asr_manager = ASRManager()
        self.llm_manager = LLMManager()
        self.tts_manager = TTSManager()
        
        # Queues for inter-task communication
        self.q_pcm = asyncio.Queue(maxsize=50)
        self.q_text = asyncio.Queue()
    
    async def check_system_status(self) -> bool:
        """Check if all system components are ready."""
        try:
            status = {
                "Audio Devices": sd.query_devices() is not None,
                "LLM Model": os.path.exists(config.paths.llm_model),
                "Output Directory": os.path.exists(config.paths.output_dir)
            }
            
            logger.info("\nSystem Status:")
            for component, is_ready in status.items():
                logger.info(f"{component}: {'OK' if is_ready else 'FAIL'}")
            
            return all(status.values())
            
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return False
    
    async def run(self):
        """Run voice interaction system."""
        if not await self.check_system_status():
            logger.error("System status check failed, cannot start")
            return
            
        logger.info("Starting voice interaction system...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            tasks = [
                self.mic_task(),
                self.asr_task(executor),
                self.llm_tts_task(executor),
            ]
            await asyncio.gather(*tasks)
    
    async def mic_task(self):
        """Microphone recording task."""
        await mic_task(self)
    
    async def asr_task(self, executor: ThreadPoolExecutor):
        """ASR processing task."""
        await asr_task(self, executor)
    
    async def llm_tts_task(self, executor: ThreadPoolExecutor):
        """LLM and TTS processing task."""
        await llm_tts_task(self, executor)
    
    def test_audio(self):
        """Test audio output."""
        return self.audio_manager.test_audio()

def validate_response_format(full_reply: str) -> str:
    """Validate and fix response format."""
    if not full_reply:
        return default_response()
        
    # Check if tags are complete
    has_think = "<think>" in full_reply and "</think>" in full_reply
    has_response = "<response>" in full_reply and "</response>" in full_reply
    
    if not (has_think and has_response):
        # Extract valid content and reformat
        content = re.sub(r'<[^>]+>', '', full_reply).strip()
        return f"""<think>
Reformatting incomplete response
</think>

<response>
{content}
</response>"""

def default_response() -> str:
    """Generate default formatted response."""
    return """<think>
Providing fallback response due to formatting issues
</think>

<response>
I apologize, but I need to format my response properly. Could you please repeat your question?
</response>"""

def quality_check(response: str) -> bool:
    """Check response quality."""
    # Check tag format
    if not (response.count("<think>") == 1 and 
            response.count("</think>") == 1 and
            response.count("<response>") == 1 and
            response.count("</response>") == 1):
        return False
            
    # Check content length
    thought, reply = ResponseProcessor.extract_thought_and_reply(response)
    
    if len(thought.split()) < 10 or len(reply.split()) < 10:
        return False
            
    # Check for duplicate content
    if ResponseProcessor.detect_content_duplication(reply):
        return False
            
    return True 