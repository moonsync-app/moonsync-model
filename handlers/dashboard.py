from dataclasses import dataclass
from typing import List, Any, Optional

import requests
import os

from datetime import datetime

import concurrent.futures
from config.base import WEATHER_API_URL


@dataclass
class DashBoardHandler:
    """
    Dashboard handler for the application.
    """

    def run_biometrics_cards(
        self,
        mood_feeling_qe: Optional[Any],
        diet_nutrition_qe: Optional[Any],
        fitness_wellness_qe: Optional[Any],
    ):
        phase = "Follicular"
        age = 35
        mood_filler = "on how the user might be feeling today. one point should suggest a way to improve mood"
        nutrition_filler = "on what the user needs to eat. one point should suggest an interesting recipe."
        exercise_filler = "on what exercises the user should perform today"

        PROMPT_TEMPLATE = """
        Current Menstrual Phase: {phase}
        Age: {age}

        Based on the above menstrual phase and other details give me 3 concise points on seperate lines (don't add index number) and nothing else {template}
        Answer in a friendly way and in second person perspective.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            mood_resp_future = executor.submit(
                mood_feeling_qe.query,
                PROMPT_TEMPLATE.format(phase=phase, age=age, template=mood_filler),
            )
            nutrition_resp_future = executor.submit(
                diet_nutrition_qe.query,
                PROMPT_TEMPLATE.format(phase=phase, age=age, template=nutrition_filler),
            )
            exercise_resp_future = executor.submit(
                fitness_wellness_qe.query,
                PROMPT_TEMPLATE.format(phase=phase, age=age, template=exercise_filler),
            )

        mood_resp = mood_resp_future.result().response
        nutrition_resp = nutrition_resp_future.result().response
        exercise_resp = exercise_resp_future.result().response

        response_json = {
            "mood_resp": mood_resp,
            "nutrition_resp": nutrition_resp,
            "exercise_resp": exercise_resp,
        }

        return response_json

    def run_header_info(
        self,
        terra_user_id: str,
        supabase: Any,
        weather_api_key: str,
        is_onboarding: Optional[bool] = False,
        terra_dev_id: Optional[str] = None,
        terra_api_key: Optional[str] = None,
    ):
        # NOTE can be changed as a catch all for biometric data from db
        if is_onboarding:
            sleep, temperature = self._get_biometric_from_terra(
                terra_user_id=terra_user_id,
                terra_api_key=terra_api_key,
                terra_dev_id=terra_dev_id,
            )
        else:
            response = (
                supabase.table("user_biometrics")
                .select("duration_in_bed_seconds_data, temperature_delta")
                .order("date", desc=True)
                .eq("terra_user_id", terra_user_id)
                .limit(1)
                .execute()
            )
            sleep, temperature = (
                response.data[0]["duration_in_bed_seconds_data"],
                98.6 + response.data[0]["temperature_delta"],
            )

        # TODO get mentrual phase from db when the model is implemented
        menstrual_phase = "Follicular"

        m, _ = divmod(sleep, 60)
        hours, mins = divmod(m, 60)

        sleep = f"{hours} hours {mins} mins"

        weather_data = self._get_weather(weather_api_key)

        response_json = {
            "menstrual_phase": menstrual_phase,
            "sleep": sleep,
            "body_temperature": round(temperature, 2),
        }
        response_json.update(weather_data)
        return response_json

    def _get_weather(self, weather_api_key):

        base_url = WEATHER_API_URL

        # TODO: get data from front end
        params = {"key": weather_api_key, "q": "New York City", "aqi": "no"}

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            location = data["location"]["name"]
            temp_f = data["current"]["temp_f"]
            condition = data["current"]["condition"]["text"]
        else:
            print("Error fetching weather data")

        return {"location": location, "condition": condition, "temp_f": temp_f}

    def _get_biometric_from_terra(
        self, terra_user_id: str, terra_dev_id: str, terra_api_key: str
    ):
        API_ROOT = "https://api.tryterra.co/v2"

        start_date = (datetime.today()).strftime("%Y-%m-%d")
        url = f"{API_ROOT}/sleep?user_id={terra_user_id}&start_date={start_date}&to_webhook=False"
        res = requests.get(
            url, headers={"dev-id": terra_dev_id, "x-api-key": terra_api_key}
        )
        print("url", url)
        sleep_data = res.json()
        print("SLEEP DATA", sleep_data)
        for data in sleep_data["data"]:
            if (
                not data["metadata"]["is_nap"]
                and data["sleep_durations_data"]["other"]["duration_in_bed_seconds"]
                > 10800
            ):
                return (
                    data["sleep_durations_data"]["other"]["duration_in_bed_seconds"],
                    98.6 + data["temperature_data"]["delta"],
                )
