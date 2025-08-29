"""Task implementations for the voice interaction system."""
import asyncio
import math
import os
import queue
import sys
import time
import logging
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import re

import numpy as np
import sounddevice as sd

from config import config
from utils import TextProcessor, AudioProcessor, ResponseProcessor

# Set up logger
logger = logging.getLogger(__name__)

async def mic_task(system):
    """Microphone recording task implementation."""
    block = int(config.audio.sample_rate * config.audio.frame_ms / 1000)
    loop = asyncio.get_running_loop()
    q_raw = queue.Queue()
    is_recording = False
    
    def _callback(indata, frames, time_, status):
        """Audio callback function."""
        if status:
            logger.error(f"Audio input error: {status}")
        current_time = time.time()
        
        # Send silence if playing, thinking or just finished playback
        if (system.state.is_playing or system.state.is_thinking or 
            current_time - system.state.last_playback_end < 2.0):  # Increased from 1.0 to 2.0 seconds
            q_raw.put(np.zeros_like(indata[:, 0]))
        else:
            q_raw.put(indata[:, 0].copy())

    try:
        # Create audio stream
        system.audio_manager.stream = sd.InputStream(
            channels=1,
            samplerate=config.audio.sample_rate,
            blocksize=block,
            callback=_callback
        )
        
        with system.audio_manager.stream:
            sentence, elapsed_ms, silent_ms = [], 0, 0
            while True:
                frame: np.ndarray = await loop.run_in_executor(None, q_raw.get)
                current_time = time.time()
                
                # Skip during playback, thinking or just after playback
                if (system.state.is_playing or system.state.is_thinking or 
                    current_time - system.state.last_playback_end < 2.0):  # Increased from 1.0 to 2.0 seconds
                    continue
                    
                energy = math.sqrt((frame ** 2).mean()) * 1000
                
                sentence.append(frame)
                elapsed_ms += config.audio.frame_ms
                
                # Update recording state
                if energy >= config.audio.energy_threshold:
                    if not is_recording:
                        logger.info("Recording started...")
                        is_recording = True
                    silent_ms = 0
                else:
                    silent_ms += config.audio.frame_ms
                    if is_recording and silent_ms >= config.audio.silence_ms:
                        is_recording = False
                        logger.info("Recording ended")

                # Process audio when enough silence or max length reached
                # Only stop if we have significant silence and some speech was recorded
                should_process = False
                
                # Check if we should process based on silence duration
                if is_recording and silent_ms >= config.audio.min_pause_between_words:
                    # We've had a pause, but check if it's long enough to stop
                    if silent_ms >= config.audio.silence_ms and elapsed_ms > config.audio.min_speech_ms:
                        should_process = True
                        logger.info(f"Processing after {silent_ms}ms silence, {elapsed_ms}ms total")
                
                # Force process if max length reached
                if elapsed_ms >= config.audio.max_sentence_ms:
                    should_process = True
                    logger.info(f"Processing due to max length: {elapsed_ms}ms")
                
                if should_process:
                    if len(sentence) > 0:
                        pcm = np.concatenate(sentence, axis=0)
                        if not AudioProcessor.detect_silence(pcm):
                            pcm = pcm.astype(np.float32)
                            await system.q_pcm.put(pcm)
                            logger.info(f"Sent {len(pcm)/config.audio.sample_rate:.1f}s of audio for processing")
                    sentence, elapsed_ms, silent_ms = [], 0, 0
                    is_recording = False
                    
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")

async def asr_task(system, executor: ThreadPoolExecutor):
    """ASR processing task implementation."""
    last_text = ""  # Store previous text
    
    while True:
        try:
            pcm = await system.q_pcm.get()
            text = await asyncio.get_running_loop().run_in_executor(
                executor, system.asr_manager.transcribe, pcm)
            
            if text:
                # Check similarity with previous text
                if last_text:
                    similarity = TextProcessor.check_similarity(last_text, text)
                    if similarity > 0.8:  # Too similar, might be duplicate
                        logger.warning(f"Detected duplicate recognition (similarity: {similarity:.2f})")
                        continue
                
                # Validate text length
                logger.info(f"[ASR] {text}")
                
                # Update previous text
                last_text = text
                
                if text.strip():
                    await system.q_text.put(text)
                    
        except Exception as e:
            logger.error(f"ASR task error: {str(e)}")

async def llm_tts_task(system, executor: ThreadPoolExecutor):
    """LLM and TTS processing task implementation."""
    consecutive_failures = 0  # Track consecutive TTS failures
    
    while True:
        try:
            user_text = await system.q_text.get()
            
            # Set thinking state, pause recording
            system.state.is_thinking = True
            logger.info("\nLLM thinking, recording paused...")
            logger.info(f"User input: {user_text}")
            
            # Keep history within reasonable limits
            if len(system.state.history) > 6:  # Reduce history length
                system.state.history = [system.state.history[0]] + system.state.history[-5:]
            
            # Add user message
            system.state.add_message("user", user_text)
            
            # Generate response
            logger.info("\nGenerating response...")
            full_reply = await asyncio.get_running_loop().run_in_executor(
                executor, system.llm_manager.generate_response, system.state.history)
            
            if full_reply:
                # Extract thought process and response separately
                think_match = re.search(r'<think>(.*?)</think>', full_reply, re.DOTALL)
                response_match = re.search(r'<response>(.*?)</response>', full_reply, re.DOTALL)
                
                # Display thought process (text only, no voice)
                if think_match:
                    think_content = think_match.group(1).strip()
                    print("\n[THOUGHT PROCESS]:")
                    print("="* 40)
                    print(think_content)
                    print("="* 40)
                
                # Display response text
                if response_match:
                    response_content = response_match.group(1).strip()
                    print("\n[RESPONSE]:")
                    print(response_content)
                    print("-"* 40)
                
                # Process response for TTS (only the response part, not the thought)
                logger.info("\nStarting speech synthesis...")
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        executor, system.tts_manager.synthesize, full_reply)
                    
                    if result and isinstance(result, tuple) and len(result) == 2:
                        logger.info("Speech synthesis successful")
                        data, fs = result
                        data = AudioProcessor.process_audio_data(data, volume_multiplier=config.audio.volume_multiplier)
                        
                        logger.info("Starting playback...")
                        system.state.is_playing = True
                        sd.play(data, fs)
                        sd.wait()
                        system.state.is_playing = False
                        system.state.last_playback_end = time.time()
                        logger.info("Playback completed")
                        consecutive_failures = 0  # Reset failure counter
                    else:
                        consecutive_failures += 1
                        logger.error(f"Speech synthesis failed (attempt {consecutive_failures}): {result}")
                        
                        # If TTS fails multiple times, continue without voice
                        if consecutive_failures >= 2:
                            logger.warning("Multiple TTS failures, continuing without voice output")
                            consecutive_failures = 0
                except Exception as tts_error:
                    logger.error(f"TTS Exception: {str(tts_error)}")
                    consecutive_failures += 1
                    
                # Always update conversation history, even if TTS fails
                system.state.add_message("assistant", full_reply)
            
            # Resume recording
            system.state.is_thinking = False
            logger.info("\nReady to listen...")
            
        except Exception as e:
            logger.error(f"LLM/TTS error: {str(e)}")
            import traceback
            logger.debug(f"Stack trace:\n{traceback.format_exc()}")
            system.state.is_thinking = False  # Ensure recording resumes
            continue