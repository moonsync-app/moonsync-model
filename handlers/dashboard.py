from dataclasses import dataclass
from typing import List, Any, Optional
import concurrent.futures


@dataclass
class DashBoardHandler:
    """
    Dashboard handler for the application.
    """

    mood_feeling_qe: Any
    diet_nutrition_qe: Any
    fitness_wellness_qe: Any

    def run(self):
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
                self.mood_feeling_qe.query,
                PROMPT_TEMPLATE.format(phase=phase, age=age, template=mood_filler),
            )
            nutrition_resp_future = executor.submit(
                self.diet_nutrition_qe.query,
                PROMPT_TEMPLATE.format(phase=phase, age=age, template=nutrition_filler),
            )
            exercise_resp_future = executor.submit(
                self.fitness_wellness_qe.query,
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
