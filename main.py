import os
import logging
from core.engine import DetectionEngine

def start_app():
    # make sure logs dir exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    print("-" * 30)
    print(" Driver Safety Monitor v1.0 ")
    print("-" * 30)
    
    try:
        # Load up the engine
        # TODO: Add dynamic config selection via CLI args later
        engine = DetectionEngine(conf_file="config.yaml")
        engine.run()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Global crash")
    finally:
        print("Goodbye!")

if __name__ == "__main__":
    start_app()
