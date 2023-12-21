from guidance import models, gen

# Create our test prompt
prompt = """
why is the sky blue?
"""
# Create our configuration for the LiteLLM endpoint. API Key is required but the value can be anything.
# Guidance explicitly requires the exact model that was supplied in the LiteLLM --model argument, you cannot use the LiteLLM --alias value
model_endpoint = models.LiteLLMCompletion(
    "ollama/openhermes2.5-mistral",
    temperature=0.8, 
    api_base="http://0.0.0.0:8000"
)

# Initiate our model endpoint, append our test prompt, and generate a response
lm = model_endpoint
lm += prompt
lm += gen()

# Convert the response to a string value and print it so we can read it
print(str(lm))
