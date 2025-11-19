import os
import csv
import random
from datetime import datetime

# Path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up from /src
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "env_dataset.csv")

 
def generate_fake_measurement():
    """
    Simulate one sensor measurement.
    Returns a dict with timestamp, temperature, humidity.
    """
    # Simulate temperature (°C): between 18 and 32
    temperature = round(random.uniform(18, 32), 2)

    # Simulate humidity (%): between 20 and 80
    humidity = round(random.uniform(20, 80), 2)

    # Current timestamp as ISO string
    timestamp = datetime.now().isoformat(timespec="seconds")

    return {
        "timestamp": timestamp,
        "temperature": temperature,
        "humidity": humidity,
    }


def init_csv_if_needed():
    """
    Create the CSV file with header if it doesn't exist.
    """
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # header
            writer.writerow(["timestamp", "temperature", "humidity"])
        print(f"Created new CSV file at: {CSV_PATH}")
    else:
        print(f"CSV file already exists at: {CSV_PATH}")


def append_measurements(n_rows=500):
    """
    Generate n_rows fake measurements and append to CSV.
    """
    init_csv_if_needed()

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for _ in range(n_rows):
            m = generate_fake_measurement()
            writer.writerow([m["timestamp"], m["temperature"], m["humidity"]])

    print(f"Added {n_rows} new measurements to {CSV_PATH}")


if __name__ == "__main__":
    # You can change this number to generate more or fewer samples
    append_measurements(n_rows=500)
