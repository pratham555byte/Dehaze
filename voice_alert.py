import queue
import threading
import time
import pyttsx3

last_alert_time = {}
ALERT_COOLDOWN = 5

# Thread-safe queue for TTS messages
_speech_queue = queue.Queue()

def _tts_worker():
    """
    Dedicated worker thread that processes text-to-speech requests sequentially.
    This prevents SAPI5 COM multi-threading collision and 'run loop already started' crashes.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except Exception as e:
        print(f"[VoiceAlert] Failed to initialize pyttsx3: {e}")
        return

    while True:
        message = _speech_queue.get()
        if message is None:
            break
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as e:
            print(f"[VoiceAlert] Error speaking: {e}")
            try:
                # Attempt to re-initialize on error
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
            except Exception:
                pass
        finally:
            _speech_queue.task_done()

# Start the worker thread
_worker_thread = threading.Thread(target=_tts_worker, daemon=True)
_worker_thread.start()

def speak_alert(alert_key, message):
    current_time = time.time()
    if alert_key in last_alert_time:
        elapsed = current_time - last_alert_time[alert_key]
        if elapsed < ALERT_COOLDOWN:
            return

    last_alert_time[alert_key] = current_time
    print(f"ALERT: {message}")
    
    # Enqueue speech task
    _speech_queue.put(message)