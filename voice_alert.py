import time
import threading
import pyttsx3

last_alert_time = {}

ALERT_COOLDOWN = 5



def _speak(message):

    engine = pyttsx3.init()

    engine.setProperty('rate', 160)

    engine.say(message)

    engine.runAndWait()


def speak_alert(alert_key, message):

    current_time = time.time()

    if alert_key in last_alert_time:

        elapsed = (
            current_time
            - last_alert_time[alert_key]
        )

        if elapsed < ALERT_COOLDOWN:
            return

    last_alert_time[alert_key] = current_time

    print(f"ALERT: {message}")

    threading.Thread(

        target=_speak,

        args=(message,),

        daemon=True

    ).start()