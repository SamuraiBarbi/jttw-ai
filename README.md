# jttw-ai

**Jesus Take the Wheel**, an A.I. assistant for people like me

## Components
- [Ollama](https://ollama.ai/)
  * What is Ollama?
    - Ollama is a local large language model server
  * [Documentation](https://github.com/jmorganca/ollama/tree/main/docs)
  * [Repository](https://github.com/jmorganca/ollama/tree/main)
  * [Discord](https://discord.com/invite/ollama)
- [LiteLLM](https://litellm.vercel.app/)
  * What is LiteLLM?
    - LiteLLM is an OpenAI proxy server
  * [Documentation](https://docs.litellm.ai/docs/proxy/quick_start)
  * [Repository](https://github.com/BerriAI/litellm/releases)
  * [Discord](https://discord.com/invite/wuPM9dRgDw)
- [Guidance](https://github.com/guidance-ai/guidance)
  * What is Guidance?
    - Guidance is a large language model prompt templating language to enforce consistant and predictable formatted responses
  * [Documentation](https://github.com/guidance-ai/guidance#example-notebooks)
  * [Example 1](https://betterprogramming.pub/a-deep-dive-into-guidances-source-code-16681a76fb20), [Example 2](https://medium.com/@akshayshinde/taking-control-of-language-models-with-microsofts-guidance-library-e711cd81654b), [Example 3](https://betterprogramming.pub/a-simple-agent-with-guidance-and-local-llm-c0865c97eaa9), [Example 4](https://www.theregister.com/2023/05/18/microsoft_guidance_project/)
  * [Repository](https://github.com/guidance-ai/guidance)
- [MemGPT](https://memgpt.ai/)
  * What is MemGPT?
    - MemGPT is a memory manager for large language models
  * [Documentation](https://memgpt.readthedocs.io/en/latest/)
  * [Repository](https://github.com/cpacker/MemGPT)
  * [Discord](https://discord.gg/9GEQrxmVyE)
- [AutoGen]
- [Aider]
- [Sweep]
- [PromptFoo]


## Install Ollama
While you can run Ollama as a docker I've found that the Ollama docker version does not work with LiteLLM since LiteLLM automatically serves Ollama rather than presenting Ollama. After determining this incompatiblity was an issue between the docker version of Ollama and LiteLLM I decided to use the sh install method for Ollama instead.
```bash
curl https://ollama.ai/install.sh | sh
```
Before launching Ollama I'm going to make sure to kill any existing processes running on port 11434 - the default port for Ollama. This is to ensure that Ollama launches with the expected port.
```bash
lsof -ti :11434 | xargs -r kill
```
## Run Ollama
``` bash
ollama serve
```
## Download Ollama Models
We'll need to download the large language models now. There's various good models to choose from which are documented at https://ollama.ai/library. For the purposes of this project we'll be using a number of different models for different purposes
Models Trained for Chatting/General Assistance
```bash
ollama run openhermes2.5-mistral
ollama run dolphin2.2-mistral:7b-q6_K
```
Models Trained for Coding/Programming
```bash
ollama run codellama
ollama run codeup 
ollama run deepseek-coder
ollama run magicoder
ollama run open-orca-platypus2
ollama run phind-codellama
ollama run starcoder
ollama run wizardcoder
```
Models Trains for SQL/Database Queries
```bash
ollama run sqlcoder
```
Models Trained for Math/Calculations
```bash
ollama run wizard-math
```
## Test Ollama
Now that we have Ollama running and serving at least one model we need to test to make sure we're it's working properly.
To test - in a new tab we're going to send a curl request to the Ollama server making sure to use one of the models we've downloaded in the "model": portion of the request. Since I've downloaded openhermes2.5-mistral that is what I'm going to specify in the "model": portion. It may take a moment but we should see activity in the tab where Ollama is running.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model": "openhermes2.5-mistral", "prompt": "Why is the sky blue?"}' http://localhost:11434/api/generate
```

## Install LiteLLM
```bash
mkdir -p $HOME/LLM/jttw/litellm/litellm_venv
cd $HOME/LLM/jttw/litellm

python3 -m venv $HOME/LLM/jttw/litellm/litellm_venv
source $HOME/LLM/jttw/litellm/litellm_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
pip3 install litellm --upgrade
pip3 install async_generator
  
```
Before launching LiteLLM I'm going to make sure to kill any existing processes running on port 8000 - the default port for LiteLLM. This is to ensure that LiteLLM launches with the expected port.
```bash
lsof -ti :8000 | xargs -r kill
```
## Run LiteLLM
```bash
litellm --model ollama/openhermes2.5-mistral --api_base http://localhost:11434 --debug
```
## Test LiteLLM
Now that we have LiteLMM running and serving as an OpenAI proxy for the Ollama endpoint we need to test to make sure we're it's working properly.
To test - in a new tab we're going to send a curl request to the LiteLLM server making sure to use one of the models we've downloaded in the "model": portion of the request. Since I've downloaded openhermes2.5-mistral that is what I'm going to specify in the "model": portion. It may take a moment but we should see activity in the tab where Ollama is running as well as tab where LiteLLM is running.
```bash
curl --location 'http://0.0.0.0:8000/chat/completions' --header 'Content-Type: application/json' --data '{"model": "ollama/openhermes2.5-mistral", "messages": [{"role": "user", "content": "why is the sky blue?"}]}'
```


## Install MemGPT
```bash
mkdir -p $HOME/LLM/jttw/memgpt/memgpt_venv
cd $HOME/LLM/jttw/memgpt/
wget -P $HOME/LLM/jttw/memgpt/ https://github.com/cpacker/MemGPT/archive/refs/heads/master.zip
unzip $HOME/LLM/jttw/memgpt/master.zip -d $HOME/LLM/jttw/memgpt/
rm $HOME/LLM/jttw/memgpt/master.zip
cp -r $HOME/LLM/jttw/memgpt/MemGPT-main/. $HOME/LLM/jttw/memgpt/
rm -r $HOME/LLM/jttw/memgpt/MemGPT-main

python3 -m venv $HOME/LLM/jttw/memgpt/memgpt_venv
source $HOME/LLM/jttw/memgpt/memgpt_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
pip3 install -e .
pip3 install transformers
pip3 install torch
export OPENAI_API_KEY=key-to-success

memgpt configure
    ? Select LLM inference provider: local
    ? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): webui
    ? Enter default endpoint: http://0.0.0.0:8000
    ? Select default model wrapper (recommended: airoboros-l2-70b-2.1): airoboros-l2-70b-2.1
    ? Select your model\'s context window (for Mistral 7B models, this is probably 8k / 8192): 8192
    ? Select embedding provider: local
    ? Select default preset: memgpt_chat
    ? Select default persona: sam_pov
    ? Select default human: basic
    ? Select storage backend for archival data: local
    Saving config to /home/owner/.memgpt/config
```

## Test MemGPT
Now that we have MemGPT running and sending requests to the LiteLLM endpoint we need to test to make sure it's working properly. We'll go through and have a conversation with the MemGPT agent.
```bash
memgpt run --model dolphin2.2-mistral:7b-q6_K --model-endpoint http://0.0.0.0:8000 --debug
```

## Install AutoGen
```bash
mkdir -p $HOME/LLM/jttw/autogen/autogen_venv
cd $HOME/LLM/jttw/autogen/
wget -P $HOME/LLM/jttw/autogen/ https://github.com/microsoft/autogen/archive/refs/heads/master.zip
unzip $HOME/LLM/jttw/autogen/master.zip -d $HOME/LLM/jttw/autogen/
rm $HOME/LLM/jttw/autogen/master.zip
cp -r $HOME/LLM/jttw/autogen/autogen-main/. $HOME/LLM/jttw/autogen/
rm -r $HOME/LLM/jttw/autogen/autogen-main

python3 -m venv $HOME/LLM/jttw/autogen/autogen_venv
source $HOME/LLM/jttw/autogen/autogen_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
pip3 install pyautogen
```
## Test AutoGen
Now that we have AutoGen in place we need to test to make sure it's working properly. We'll create a python script in our $HOME/LLM/jttw/autogen/ directory named autogen_litellm_test.py.
```python
from autogen import UserProxyAgent, ConversableAgent, config_list_from_json

def main():
    # Load LLM inference endpoints from an env variable or a file
    # See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
    # and OAI_CONFIG_LIST_sample.
    # For example, if you have created an OAI_CONFIG_LIST file in the current working directory, that file will be used.

    config_list = [
        {
            "model": "ollama/openhermes2.5-mistral",
            "base_url": "http://0.0.0.0:8000",
            "api_key": "key-to-success"
        }
    ]
    # Create the agent that uses the LLM.
    assistant = ConversableAgent("agent", llm_config={"config_list": config_list})

    # Create the agent that represents the user in the conversation.
    user_proxy = UserProxyAgent("user", code_execution_config=False)

    # Let the assistant start the conversation. It will end when the user types exit.
    assistant.initiate_chat(user_proxy, message="How can I help you today?")


if __name__ == "__main__":
    main()
```
Next we'll enter our python environment for AutoGen and run the python test script. If everything is working properly we should be having a conversation with our LLM being served from Autogen through LiteLLM to Ollama.
```bash
cd $HOME/LLM/jttw/autogen/
source $HOME/LLM/jttw/autogen/autogen_venv/bin/activate
python3 autogen_litellm_test.py
```

