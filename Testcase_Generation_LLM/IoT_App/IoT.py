import datetime
import time
import random

# Simulated sensors and actuators
def get_current_temperature():
    """Simulate reading the current room temperature."""
    return random.uniform(5, 30)  # Random temperature between 5 and 30 degrees Celsius

def turn_on_light():
    print("Lights turned ON.")

def turn_off_light():
    print("Lights turned OFF.")

def turn_on_ac():
    print("AC turned ON.")

def turn_on_heater():
    print("AC switched to Heater mode.")

def turn_off_ac():
    print("AC turned OFF.")

def turn_on_fan():
    print("Fan turned ON.")

def turn_off_fan():
    print("Fan turned OFF.")

def start_washing_machine():
    print("Washing machine started.")

def water_garden():
    print("Garden watering started.")

# Main automation logic
def home_automation_system():
    while True:
        now = datetime.datetime.now()
        print(now)
        current_hour = now.hour
        current_minute = now.minute
        current_day = now.weekday()  # 0 = Monday, 6 = Sunday
        current_month = now.month

        # Lights control
        if 18 <= current_hour or current_hour < 5:  # Between 6 PM and 5 AM
            turn_on_light()
        else:
            turn_off_light()

        # Temperature control
        current_temperature = get_current_temperature()
        print(f"Current Temperature : {current_temperature}")
        if current_temperature > 24:
            turn_on_ac()
            if 4 <= current_month <= 6:  # April to June
                turn_on_fan()
            else:
                turn_off_fan()
        elif current_temperature < 10:
            turn_on_heater()
        else:
            turn_off_ac()
            turn_off_fan()

        # Washing machine control
        if current_day == 6 and current_hour == 10 and current_minute == 0:  # Sunday at 10:00 AM
            start_washing_machine()

        # Garden watering control
        if current_hour == 7 and current_minute == 0:  # Every day at 7:00 AM
            water_garden()

        # Wait for 1 minute before the next check
        time.sleep(60)

if __name__ == "__main__":
    print("Starting Home IoT Automation System...")
    try:
        home_automation_system()
    except KeyboardInterrupt:
        print("Home IoT Automation System stopped.")
