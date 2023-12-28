import os
from langchain.chat_models import ChatLiteLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatLiteLLM(
        api_base="http://0.0.0.0:11434",
        model="ollama/openhermes2.5-mistral"
)
messages = [
    HumanMessage(
        content="why is the sky blue?"
    )
]

response = chat(messages)

print(response)
