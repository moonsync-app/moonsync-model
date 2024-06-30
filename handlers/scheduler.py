from dataclasses import dataclass
from typing import List, Any
from llama_index.tools.google import GoogleCalendarToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage, MessageRole


@dataclass
class EventSchedulerHandler:
    """
    Event scheduler handler used in the application.
    """

    prompt: str
    messages: List[Any]
    llm: Any
    current_date: str

    def run(self):
        curr_history = []
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            curr_history.append(ChatMessage(role=role, content=content))

        curr_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"Very important - The timezone is EST (UTC−05:00) and the location is New York City. Current Date: {self.current_date}",
            )
        )
        tool_spec = GoogleCalendarToolSpec()
        self.agent = OpenAIAgent.from_tools(
            tool_spec.to_tool_list(),
            verbose=True,
            llm=self.llm,
            chat_history=curr_history,
        )
        response = self.agent.stream_chat(self.prompt)
        self.agent.reset()
        response_gen = response.response_gen

        return response_gen
