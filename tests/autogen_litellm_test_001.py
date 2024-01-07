from autogen import UserProxyAgent, ConversableAgent

# Create our configuration for the LiteLLM endpoint. API Key is required but the value can be anything. Set model to any model that we've downloaded 
config_list = [
    {
        "base_url": "http://0.0.0.0:8000",
        "api_key": "key-to-success",
        "model": "openhermes2.5-mistral"        
    }
]

# Create the agent that uses the LLM.
assistant = ConversableAgent(
    "agent", 
    llm_config = {
        "config_list": config_list
    }
)

# Create the agent that represents the user in the conversation.
user_proxy = UserProxyAgent(
    "user", 
    code_execution_config = False
)

# Let the assistant start the conversation.  It will end when the user types exit.
assistant.initiate_chat(
    user_proxy, 
    message = "How can I help you today?"
)
