from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import requests
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OnboardingHandler:
    terra_user_id: str
    terra_dev_id: str
    terra_api_key: str
    supabase: Any

    def run(self):
        biometric_data = {}
        all_data_params = [
            "date",
            "terra_user_id",
            "avg_hr_bpm",
            "resting_hr_bpm",
            "duration_in_bed_seconds_data",
            "duration_deep_sleep",
            "temperature_delta",
            "avg_saturation_percentage",
            "recovery_score",
            "activity_score",
            "sleep_score",
            "stress_data",
            "number_steps",
            "total_burned_calories",
        ]

        for i in range(91):
            date = (datetime.today() - timedelta(i)).strftime("%Y-%m-%d")
            biometric_data[date] = {key: None for key in all_data_params}
            biometric_data[date]["date"] = date
            biometric_data[date]["terra_user_id"] = self.terra_user_id

        API_ROOT = "https://api.tryterra.co/v2"
        DEFAULT_START_DATE = (datetime.today() - timedelta(30)).strftime("%Y-%m-%d")
        DEFAULT_END_DATE = datetime.today().strftime("%Y-%m-%d")

        def get_biometric_data(
            user_id,
            category,
            end_date=DEFAULT_END_DATE,
            start_date=DEFAULT_START_DATE,
            to_webhook=False,
        ):
            url = f"{API_ROOT}/{category}?user_id={user_id}&start_date={start_date}&end_date={end_date}&to_webhook={to_webhook}"
            res = requests.get(
                url,
                headers={"dev-id": self.terra_dev_id, "x-api-key": self.terra_api_key},
            )
            return res.json()

        parameters = [
            (
                self.terra_user_id,
                "sleep",
                (datetime.today()).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(30)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "sleep",
                (datetime.today() - timedelta(31)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(60)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "sleep",
                (datetime.today() - timedelta(61)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(90)).strftime("%Y-%m-%d"),
            ),
        ]

        # SLEEP DATA
        with ThreadPoolExecutor(max_workers=12) as executor:
            sleep_data1, sleep_data2, sleep_data3 = executor.map(
                lambda params: get_biometric_data(*params), parameters
            )
            print(sleep_data1.keys(), sleep_data2.keys(), sleep_data3.keys())

        total_sleep_data = (
            sleep_data3["data"] + sleep_data2["data"] + sleep_data1["data"]
        )
        for data in total_sleep_data:
            if (
                not data["metadata"]["is_nap"]
                and data["sleep_durations_data"]["other"]["duration_in_bed_seconds"]
                > 10800
            ):
                end_date_sliced = data["metadata"]["end_time"][:10]
                if end_date_sliced in biometric_data:
                    biometric_data[end_date_sliced]["date"] = end_date_sliced
                    biometric_data[end_date_sliced]["terra_user_id"] = self.terra_user_id
                    biometric_data[end_date_sliced]["avg_hr_bpm"] = data[
                        "heart_rate_data"
                    ]["summary"]["avg_hr_bpm"]
                    biometric_data[end_date_sliced]["resting_hr_bpm"] = data[
                        "heart_rate_data"
                    ]["summary"]["resting_hr_bpm"]
                    biometric_data[end_date_sliced]["duration_in_bed_seconds_data"] = (
                        data["sleep_durations_data"]["other"]["duration_in_bed_seconds"]
                    )
                    biometric_data[end_date_sliced]["duration_deep_sleep"] = data[
                        "sleep_durations_data"
                    ]["asleep"]["duration_deep_sleep_state_seconds"]
                    biometric_data[end_date_sliced]["temperature_delta"] = data[
                        "temperature_data"
                    ]["delta"]

        logger.info("[BIOMETRICS] Sleep data processed")

        parameters = [
            (
                self.terra_user_id,
                "body",
                (datetime.today()).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(30)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "body",
                (datetime.today() - timedelta(31)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(60)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "body",
                (datetime.today() - timedelta(61)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(90)).strftime("%Y-%m-%d"),
            ),
        ]

        # BODY DATA
        with ThreadPoolExecutor() as executor:
            body_data1, body_data2, body_data3 = executor.map(
                lambda params: get_biometric_data(*params), parameters
            )

        total_body_data = body_data3["data"] + body_data2["data"] + body_data1["data"]
        for data in total_body_data:
            end_date_sliced = data["metadata"]["end_time"][:10]
            if end_date_sliced in biometric_data:
                biometric_data[end_date_sliced]["avg_saturation_percentage"] = data[
                    "oxygen_data"
                ]["avg_saturation_percentage"]

        logger.info("[BIOMETRICS] Body data processed")

        # DAILY DATA

        parameters = [
            (
                self.terra_user_id,
                "daily",
                (datetime.today() - timedelta(1)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(30)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "daily",
                (datetime.today() - timedelta(31)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(60)).strftime("%Y-%m-%d"),
            ),
            (
                self.terra_user_id,
                "daily",
                (datetime.today() - timedelta(61)).strftime("%Y-%m-%d"),
                (datetime.today() - timedelta(90)).strftime("%Y-%m-%d"),
            ),
        ]

        with ThreadPoolExecutor() as executor:
            daily_data1, daily_data2, daily_data3 = executor.map(
                lambda params: get_biometric_data(*params), parameters
            )

        total_daily_data = (
            daily_data3["data"] + daily_data2["data"] + daily_data1["data"]
        )
        for data in total_daily_data:
            end_date_sliced = data["metadata"]["end_time"][:10]
            if end_date_sliced in biometric_data:
                biometric_data[end_date_sliced]["recovery_score"] = data["scores"][
                    "recovery"
                ]
                biometric_data[end_date_sliced]["activity_score"] = data["scores"][
                    "activity"
                ]
                biometric_data[end_date_sliced]["sleep_score"] = data["scores"]["sleep"]
                biometric_data[end_date_sliced]["stress_data"] = data["stress_data"][
                    "rest_stress_duration_seconds"
                ]
                biometric_data[end_date_sliced]["number_steps"] = data["distance_data"][
                    "steps"
                ]
                biometric_data[end_date_sliced]["total_burned_calories"] = data[
                    "calories_data"
                ]["total_burned_calories"]

        logger.info("[BIOMETRICS] Daily data processed")

        data = list(biometric_data.values())
        response = self.supabase.table("user_biometrics").insert(data).execute()
        process_response = (
            self.supabase.table("users")
            .update({"initial_load_complete": True})
            .eq("terra_user_id", self.terra_user_id)
            .execute()
        )

        logger.info("[SUPABASE INSERT] Data inserted into user_biometrics table")

        return biometric_data
