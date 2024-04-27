#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
from terra.base_client import Terra
import os
import pandas as pd
import requests

TERRA_API_KEY = os.getenv("TERRA_API_KEY")
TERRA_DEV_ID = os.getenv("TERRA_DEV_ID")
TERRA_SECRET = os.getenv("TERRA_SECRET")

terra = Terra(TERRA_API_KEY, TERRA_DEV_ID, TERRA_SECRET)


TERRA_API_ROOT = "https://api.tryterra.co/v2"
TERRA_USER_ID = "688c8437-2d16-48de-b659-48317a63edb8"

DEFAULT_START_DATE = (datetime.today() - timedelta(28)).strftime("%Y-%m-%d")
DEFAULT_END_DATE = datetime.today().strftime("%Y-%m-%d")


def get_biometric_data(
    user_id,
    category,
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    to_webhook=False,
):
    url = f"{TERRA_API_ROOT}/{category}?user_id={user_id}&start_date={start_date}&end_date={end_date}&to_webhook={to_webhook}"
    res = requests.get(
        url, headers={"dev-id": TERRA_DEV_ID, "x-api-key": TERRA_API_KEY}
    )
    return res.json()


sleep_data_4_weeks = get_biometric_data(
    TERRA_USER_ID,
    "sleep",
    start_date=(datetime.today() - timedelta(28)).strftime("%Y-%m-%d"),
)

avg_hr_bpm = []
resting_hr_bpm = []
duration_in_bed_seconds_data = []
duration_deep_sleep = []
temperature_delta = []
end_date = []
for data in sleep_data_4_weeks["data"]:
    if not data["metadata"]["is_nap"]:
        print(data["metadata"]["start_time"], data["metadata"]["end_time"])
        end_date.append(data["metadata"]["end_time"])
        avg_hr_bpm.append(data["heart_rate_data"]["summary"]["avg_hr_bpm"])
        resting_hr_bpm.append(data["heart_rate_data"]["summary"]["resting_hr_bpm"])
        duration_in_bed_seconds_data.append(
            data["sleep_durations_data"]["other"]["duration_in_bed_seconds"]
        )
        duration_deep_sleep.append(
            data["sleep_durations_data"]["asleep"]["duration_deep_sleep_state_seconds"]
        )
        temperature_delta.append(data["temperature_data"]["delta"])

print("Last 4 weeks Avg HR BPM: ", avg_hr_bpm, len(avg_hr_bpm))
print("Last 4 weeks Resting HR BPM: ", resting_hr_bpm, len(resting_hr_bpm))
print(
    "Last 4 weeks Sleep Duration Data: ",
    duration_in_bed_seconds_data,
    len(duration_in_bed_seconds_data),
)
print(
    "Last 4 weeks Deep Sleep Duration Data: ",
    duration_deep_sleep,
    len(duration_deep_sleep),
)
print(
    "Last 4 weeks Temperature Delta Data: ", temperature_delta, len(temperature_delta)
)

body_data_4_weeks = get_biometric_data(
    TERRA_USER_ID,
    "body",
    start_date=(datetime.today() - timedelta(29)).strftime("%Y-%m-%d"),
)

avg_saturation_percentage = []
for data in body_data_4_weeks["data"]:
    print(data["metadata"]["start_time"], data["metadata"]["end_time"])
    avg_saturation_percentage.append(data["oxygen_data"]["avg_saturation_percentage"])

print(
    "Last 4 weeks Oxygen Data: ",
    avg_saturation_percentage,
    len(avg_saturation_percentage),
)

daily_data = get_biometric_data(
    TERRA_USER_ID,
    "daily",
    start_date=(datetime.today() - timedelta(27)).strftime("%Y-%m-%d"),
)

recovery_score = []
activity_score = []
sleep_score = []
stress_data = []
number_steps = []
total_burned_calories = []

for i in range(len(daily_data["data"])):
    print(
        daily_data["data"][i]["metadata"]["start_time"],
        daily_data["data"][i]["metadata"]["end_time"],
    )
    recovery_score.append(daily_data["data"][i]["scores"]["recovery"])
    activity_score.append(daily_data["data"][i]["scores"]["activity"])
    sleep_score.append(daily_data["data"][i]["scores"]["sleep"])
    stress_data.append(
        daily_data["data"][i]["stress_data"]["rest_stress_duration_seconds"]
    )
    number_steps.append(daily_data["data"][i]["distance_data"]["steps"])
    total_burned_calories.append(
        daily_data["data"][i]["calories_data"]["total_burned_calories"]
    )

print("Last 4 weeks Recovery Score: ", recovery_score, len(recovery_score))
print("Last 4 weeks Activity Score: ", activity_score, len(activity_score))
print("Last 4 weeks Sleep Score: ", sleep_score, len(sleep_score))
print("Last 4 weeks Stress Data: ", stress_data, len(stress_data))
print("Last 4 weeks Number of Steps: ", number_steps, len(number_steps))
print(
    "Last 4 weeks Net Burned Calories: ",
    total_burned_calories,
    len(total_burned_calories),
)

column_names = [
    "date",
    "recovery_score",
    "activity_score",
    "sleep_score",
    "stress_data",
    "number_steps",
    "total_burned_calories",
    "avg_saturation_percentage",
    "avg_hr_bpm",
    "resting_hr_bpm",
    "duration_in_bed_seconds_data",
    "duration_deep_sleep",
    "temperature_delta",
]

df = pd.DataFrame(columns=column_names)

# fill the dataframe with the right data lists
df["date"] = end_date
df["recovery_score"] = recovery_score
df["activity_score"] = activity_score
df["sleep_score"] = sleep_score
df["stress_data"] = stress_data
df["number_steps"] = number_steps
df["total_burned_calories"] = total_burned_calories
df["avg_saturation_percentage"] = avg_saturation_percentage
df["avg_hr_bpm"] = avg_hr_bpm
df["resting_hr_bpm"] = resting_hr_bpm
# df['oxygen_saturation_data'] = oxygen_saturation_data
df["duration_in_bed_seconds_data"] = duration_in_bed_seconds_data
df["duration_deep_sleep"] = duration_deep_sleep
df["temperature_delta"] = temperature_delta

df.to_csv("biometric_data.csv", index=False)
