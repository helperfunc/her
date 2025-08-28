"""Main entry point for the voice interaction system."""
import asyncio
import sys
import logging
from pathlib import Path

from voice_system import VoiceSystem
from config import config

logger = logging.getLogger(__name__)

async def setup_system() -> VoiceSystem:
    """Initialize voice interaction system."""
    try:
        logger.info("Initializing voice interaction system...")
        system = VoiceSystem()
        
        # Check system status
        if not await system.check_system_status():
            raise RuntimeError("System status check failed")
            
        # Test audio setup
        if not system.test_audio():
            raise RuntimeError("Audio test failed")
            
        logger.info("System initialization completed")
        return system
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

async def run_system(system: VoiceSystem):
    """Run voice interaction system."""
    try:
        logger.info("Starting voice interaction system")
        await system.run()
    except Exception as e:
        logger.error(f"System runtime error: {str(e)}")
        raise

async def main_async():
    """Asynchronous main function."""
    try:
        # Initialize system
        system = await setup_system()
        
        # Run system
        await run_system(system)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down system")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

def main():
    """Main entry function."""
    try:
        # Run async main function
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nSystem shutdown (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\nSystem error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 