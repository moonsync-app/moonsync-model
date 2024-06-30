from dataclasses import dataclass
from typing import List, Any, Optional
from llama_index.core.llms import ChatMessage, MessageRole

from config.prompts import FORWARD_PROMPT, SYSTEM_PROMPT
from models import CustomCondenseQuestionChatEngine


@dataclass
class ChatHandler:
    prompt: str
    messages: List[Any]
    user_info_content: Optional[str] = None
    sub_question_query_engine: Optional[Any] = None
    chat_engine: Optional[Any] = None
    menstrual_phase_info: Optional[str] = None

    def run_offline(self):
        print("Prompt: ", self.prompt)
        if len(self.messages) == 0:
            self.prompt = self.prompt + "\n" + self.menstrual_phase_info

        print("incoming messages", self.messages)  # role and content
        if len(self.messages) > 0:
            self.chat_engine.reset()
            curr_history = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=self.content_template,
                ),
            ]
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                curr_history.append(ChatMessage(role=role, content=content))

            self.chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
                query_engine=self.sub_question_query_engine,
                condense_question_prompt=FORWARD_PROMPT,
                chat_history=curr_history,
                verbose=True,
            )
        streaming_response = self.chat_engine.stream_chat(self.prompt)
        self.chat_engine.reset()
        chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=self.user_info_content,
            ),
        ]
        self.chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
            query_engine=self.sub_question_query_engine,
            condense_question_prompt=FORWARD_PROMPT,
            chat_history=chat_history,
            verbose=True,
        )

        return self.chat_engine, streaming_response

    def run_online(self):
        print("Prompt: ", self.prompt)
        self.prompt = self.prompt.replace("@internet", "")

        # TODO change current location
        curr_history = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            curr_history.append(ChatMessage(role=role, content=content))

        curr_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content=self.prompt
                + "\n"
                + self.user_info_content
                + "\nGive the output in a markdown format and ask the user if they want to schedule the event if relevant to the context. STRICTLY FOLLOW - Give a short and concise answer.",
            )
        )

        return curr_history
