"""
violation_logger.py
Reusable violation-logging helper with cooldown-based deduplication,
for use across detection apps (PPE Detector, Traffic Violation Detector).
"""

import csv
import os
import time
from datetime import datetime


class ViolationLogger:
    def __init__(self, csv_path, cooldown_seconds = 5):
        self.csv_path = csv_path
        self.cooldown_seconds = cooldown_seconds
        self.last_logged_time = {}

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "violation", "severity"])

    def log(self, violation_key, severity):
        now = time.time()
        last_time = self.last_logged_time.get(violation_key, 0)

        if now - last_time > self.cooldown_seconds:
            self.last_logged_time[violation_key] = now
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, violation_key, severity])
            return True  # actually logged
        return False  # skipped due to cooldown