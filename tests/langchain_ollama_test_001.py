from langchain.chat_models import ChatLiteLLM
from langchain.schema import HumanMessage

chat = ChatLiteLLM(
        api_base="http://0.0.0.0:11434",
        model="ollama/openhermes2.5-mistral",
)

messages = [
    HumanMessage(
        content="why is the sky blue?"
    )
]

response = chat(messages)

print(response)
