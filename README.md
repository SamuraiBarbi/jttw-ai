# jttw-ai

**Jesus Take the Wheel**, an A.I. assistant for people like me

## Terms To Know
| Term                             | Definition                                                                                                                                         |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Prompt**                       | A prompt is a user-provided input or instruction given to an AI system to generate a specific response or perform a task.                           |
| **Inference**                    | Inference in AI refers to the process of using a trained model to make predictions or generate outputs based on new, unseen data.                      |
| **Model**                        | A model is a mathematical representation or framework that an AI system uses to learn patterns from data and make predictions or generate responses.  |
| **LLM (Language Model)**         | A Language Model (LLM) is a type of AI model specifically designed for understanding and generating human language.                                   |
| **Tokens**                       | Tokens are units of input text that a language model processes. They can be as short as a single character or as long as an entire word.               |
| **Context**                      | Context refers to the information or surroundings that influence the interpretation of a given input or task in AI systems.                          |
| **Chat, Instruct, Completion**   | These terms represent different types of tasks that AI models can perform, such as engaging in conversation (Chat), following instructions (Instruct), or completing a given prompt (Completion). |
| **GPU (Graphics Processing Unit)** | A GPU is a hardware component that accelerates the training and inference processes of AI models by parallelizing computations.                       |
| **CPU (Central Processing Unit)** | A CPU is a general-purpose processor that handles tasks related to overall system management and execution of instructions in AI applications.        |
| **RAM (Random Access Memory)**    | RAM is a type of computer memory that provides fast and temporary storage for data that the CPU is currently using or processing.                        |
| **VRAM (Video Random Access Memory)** | VRAM is a specific type of RAM used by GPUs to store graphical data, such as textures and frame buffers, for rendering images.                          |
| **CUDA Cores**                   | CUDA Cores are processing units within a GPU that handle parallel computations, commonly used in deep learning tasks.                                  |
| **Tensor Cores**                 | Tensor Cores are specialized processing units in modern GPUs designed for accelerating matrix operations, particularly beneficial for deep learning workloads. |
| **OOM (Out of Memory)**           | OOM occurs when a program or process exhausts the available memory and is unable to allocate more, leading to system instability or termination.        |
| **Quantize**                     | Quantization is a technique in AI model optimization that involves reducing the precision of numerical values, often to lower bit-widths, to decrease memory and computation requirements. |

## Required Reading
  * ### Prompt Engineering
    We'll need to get up to speed with best practices for writing prompts that produce the intended responses from large language models ( LLM ). The best resource for this is the [Awesome GPT Prompt Engineering](https://github.com/snwfdhmp/awesome-gpt-prompt-engineering) repo. It's comprehensive collection of the best prompt engineering guides and information currated by the A.I. entheusiasts community. Read it.
  * ### Hardware Requirements
    In order to run LLM locally hosted on our own machine/server we'll need to understand the hardware requirements, costs, and best available hardware options in accordance to our budget. The best resource for this is Tim Dettmers' comprehensive write up on [The Best GPUs for Deep Learning in 2023](https://github.com/snwfdhmp/awesome-gpt-prompt-engineering). Read it. If you can't be bothered to read it then refer to his charts [Best GPUs by Relative Performance](https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/GPUS_Ada_raw_performance3.png?ssl=1), [Best GPUs by Relative Performance Per Dollar](https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/GPUs_Ada_performance_per_dollar6.png?ssl=1), and [GPU Recommendation Chart](https://i0.wp.com/timdettmers.com/wp-content/uploads/2023/01/gpu_recommendations.png?ssl=1).

    Tim's post specifically covers loading LLM into GPU memory ( VRAM ). He does not get into loading LLM into shared memory like MacBook Pros. Generally as of the time of me writing this the only methods of loading LLM and having fast responses are using GPU or certain models of MacBook that have large amounts of memory. While we can load LLM with system memory ( RAM ) and run them on the systems CPU, it will be incredibly slow compared to the other two methods.
    
  
## Components
- ### [Ollama](https://ollama.ai/)
  * What is Ollama?
    - Ollama is a local large language model server.
  * [Installation](#install-ollama), [Running](#run-ollama), [Downloading Models](#download-ollama-models), [Testing](#test-ollama)
  * [Documentation](https://github.com/jmorganca/ollama/tree/main/docs)
  * [Repository](https://github.com/jmorganca/ollama/tree/main)
  * [Discord](https://discord.com/invite/ollama)
- ### [LiteLLM](https://litellm.vercel.app/)
  * What is LiteLLM?
    - LiteLLM is an OpenAI proxy server.
  * [Installation](#install-litellm), [Running](#run-litellm), [Testing](#test-litellm)
  * [Documentation](https://docs.litellm.ai/docs/proxy/quick_start)
  * [Repository](https://github.com/BerriAI/litellm/releases)
  * [Discord](https://discord.com/invite/wuPM9dRgDw)
- ### [Guidance](https://github.com/guidance-ai/guidance)
  * What is Guidance?
    - Guidance is a large language model prompt templating language to enforce consistant and predictable formatted responses.
  * [Installation](#install-guidance), [Testing](#test-guidance)
  * [Documentation](https://github.com/guidance-ai/guidance#example-notebooks)
  * [Example 1](https://betterprogramming.pub/a-deep-dive-into-guidances-source-code-16681a76fb20), [Example 2](https://medium.com/@akshayshinde/taking-control-of-language-models-with-microsofts-guidance-library-e711cd81654b), [Example 3](https://betterprogramming.pub/a-simple-agent-with-guidance-and-local-llm-c0865c97eaa9), [Example 4](https://www.theregister.com/2023/05/18/microsoft_guidance_project/)
  * [Repository](https://github.com/guidance-ai/guidance)
- ### [MemGPT](https://memgpt.ai/)
  * What is MemGPT?
    - MemGPT is a memory manager for LLM that facilitates the ability to recall/remember information that well exceeds typical context length limits.
  * [Installation](#install-memgpt), [Configuring](#configure-memgpt), [Testing](#test-memgpt)
  * [Documentation](https://memgpt.readthedocs.io/en/latest/)
  * [Repository](https://github.com/cpacker/MemGPT)
  * [Discord](https://discord.gg/9GEQrxmVyE)
- ### [AutoGen](https://microsoft.github.io/autogen/)
  * What is AutoGen?
    - Autogen is a multi-agent framework for directing multiple LLM to work together to complete tasks given to them as a group.
  * [Installation](#install-autogen), [Testing](#test-autogen)
  * [Documentation](https://microsoft.github.io/autogen/docs/Getting-Started)
  * [Example 1](https://microsoft.github.io/autogen/docs/Examples)
  * [Repository](https://github.com/microsoft/autogen)
  * [Discord](https://discord.gg/pAbnFJrkgZ)
- ### [Aider](https://aider.chat/)
  * What is Aider?
    - Aider is command line tool for instructing LLM to create, update, revise, improve, and document/comment code for local git repositories
  * [Installation](#install-aider)
  * [Documentation](https://aider.chat/examples/)
  * [Respository](https://github.com/paul-gauthier/aider)
- ### [Sweep]
- ### [PromptFoo](https://www.promptfoo.dev/)
  * What is PromptFoo?
    - PromptFoo is a unit testing and evaluation tool for prompts allowing for a better understand and insight into prompts that consistantly behave as expected, and those that need refactored.
  * [Installation](#install-promptfoo)
  * [Documentation](https://www.promptfoo.dev/docs/intro/)
  * [Repository](https://github.com/promptfoo/promptfoo)
  * [Discord](https://discord.gg/gHPS9jjfbs)


## Install Ollama
While you can run Ollama as a docker I've found that the Ollama docker version does not work with LiteLLM since LiteLLM automatically serves Ollama rather than presenting Ollama. After determining this incompatiblity was an issue between the docker version of Ollama and LiteLLM I decided to use the sh install method for Ollama instead. Your mileage may vary if you instead decide to use the Ollama docker version as well as [the Ollama compatible LiteLLM docker](https://litellm.vercel.app/docs/providers/ollama#quick-start).
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
We'll need to download the LLM now. There's various good models to choose from which are documented at https://ollama.ai/library. For the purposes of this project we'll be using a number of different models for different purposes

Models Trained for Chatting/General Assistance
```bash
ollama run openhermes2.5-mistral
ollama run dolphin2.2-mistral:7b-q6_K
ollama run dolphin-mixtral
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
Models Trained for SQL/Database Queries
```bash
ollama run sqlcoder
```
Models Trained for Math/Calculations
```bash
ollama run wizard-math
```
Models Trained for Image Analysis
```bash
ollama run llava
```
## Test Ollama
Now that we have Ollama running and serving at least one model we need to test to make sure we're it's working properly.
To test - in a new tab we're going to send a curl request to the Ollama server making sure to use one of the models we've downloaded in the "model": portion of the request. Since I've downloaded openhermes2.5-mistral that is what I'm going to specify in the "model": portion. It may take a moment but we should see activity in the tab where Ollama is running.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"model": "openhermes2.5-mistral", "prompt": "Why is the sky blue?"}' http://localhost:11434/api/generate
```

## Install LiteLLM
We're installing specifically the litellm 1.14.1 version of the litellm python package because latest verion 1.15.1 introduced a [bug that broke Ollama requests](https://github.com/BerriAI/litellm/issues/1156). The fix should be out soon in the next release through.
```bash
# Create our the directory for our LiteLLM python environment
mkdir -p $HOME/LLM/jttw/litellm/litellm_venv
cd $HOME/LLM/jttw/litellm

# Create our LiteLLM python environment
python3 -m venv $HOME/LLM/jttw/litellm/litellm_venv
# Activate into it, install/update pip, and clear python package cache
source $HOME/LLM/jttw/litellm/litellm_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
# Install LiteLLM python package and the required Async Generator python package
pip3 install litellm==1.14.1
pip3 install async_generator
  
```
Before launching LiteLLM I'm going to make sure to kill any existing processes running on port 8000 - the default port for LiteLLM. This is to ensure that LiteLLM launches with the expected port.
```bash
lsof -ti :8000 | xargs -r kill
```
## Run LiteLLM
We are adding the --debug argument so that we can we detailed info about requests LiteLLM receives and it's responses back. We're also adding the [--add_function_to_prompt](https://docs.litellm.ai/docs/proxy/cli#--add_function_to_prompt) and [--drop_params](https://docs.litellm.ai/docs/proxy/cli#--drop_params) arguments when executing litellm because MemGPT makes heavy use of function calling and params, which Ollama does not support and without these arguments LiteLLM will produce errors indicating that Ollama does not support the function and param calls MemGPT is sending.
```bash
litellm --model ollama/openhermes2.5-mistral --api_base http://localhost:11434 --debug --add_function_to_prompt --drop_params
```
## Test LiteLLM
Now that we have LiteLMM running and serving as an OpenAI proxy for the Ollama endpoint we need to test to make sure we're it's working properly.
To test - in a new tab we're going to send a curl request to the LiteLLM server making sure to use one of the models we've downloaded in the "model": portion of the request. Since I've downloaded openhermes2.5-mistral that is what I'm going to specify in the "model": portion. It may take a moment but we should see activity in the tab where Ollama is running as well as tab where LiteLLM is running.
```bash
curl --location 'http://0.0.0.0:8000/chat/completions' --header 'Content-Type: application/json' --data '{"model": "ollama/openhermes2.5-mistral", "messages": [{"role": "user", "content": "why is the sky blue?"}]}'
```
## Install Guidance
```bash
# Create our the directory for our Guidance python environment
mkdir -p $HOME/LLM/jttw/guidance/guidance_venv
cd $HOME/LLM/jttw/guidance/

# Create our Guidance python environment
python3 -m venv $HOME/LLM/jttw/guidance/guidance_venv
# Activate into it, install/update pip, clear python package cache
source $HOME/LLM/jttw/guidance/guidance_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
# Install Guidance python package and the required LiteLLM python package
pip3 install guidance --upgrade
pip3 install litellm==1.14.1
```
## Test Guidance
Now that we have Guidance in place we need to test to make sure it's working properly. We'll create a python script in our $HOME/LLM/jttw/guidance/ directory and name the file guidance_litellm_test.py. Fun fact, the LiteLLM model is not called in Guidance like other models traditionally would be - example models.OpenAI, but rather by one of three methods as detailed in the [Guidance _lite_llm.py class in their repo](https://github.com/guidance-ai/guidance/blob/main/guidance/models/_lite_llm.py) - LiteLLMChat, LiteLLMInstruct, or LiteLLMCompletion. For this test we're going to use LiteLLMCompletion.
```python
from guidance import models, gen

prompt = """
why is the sky blue?
"""

model_endpoint = models.LiteLLMCompletion(
    "ollama/openhermes2.5-mistral",
    temperature=0.8, 
    api_base="http://0.0.0.0:8000"
)

lm = model_endpoint
lm += prompt
lm += gen()

print(str(lm))
```
Next we'll enter our python environment for Guidance and run the python test script. If everything is working properly we should be having output from our LLM being served from Guidance through LiteLLM to Ollama.
```bash
cd $HOME/LLM/jttw/guidance/
source $HOME/LLM/jttw/guidance/guidance_venv/bin/activate
python3 guidance_litellm_test.py
```

## Install MemGPT
```bash
# Create our the directory for our MemGPT python environment
mkdir -p $HOME/LLM/jttw/memgpt/memgpt_venv
cd $HOME/LLM/jttw/memgpt/

wget -P $HOME/LLM/jttw/memgpt/ https://github.com/cpacker/MemGPT/archive/refs/heads/master.zip
unzip $HOME/LLM/jttw/memgpt/master.zip -d $HOME/LLM/jttw/memgpt/
rm $HOME/LLM/jttw/memgpt/master.zip
cp -r $HOME/LLM/jttw/memgpt/MemGPT-main/. $HOME/LLM/jttw/memgpt/
rm -r $HOME/LLM/jttw/memgpt/MemGPT-main

# Create our MemGPT python environment
python3 -m venv $HOME/LLM/jttw/memgpt/memgpt_venv
# Activate into it, install/update pip, clear python package cache
source $HOME/LLM/jttw/memgpt/memgpt_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
# Install MemGPT python package and the required Transformers and PyTorch python packages
pip3 install -e .
pip3 install transformers
pip3 install torch

```
## Configure MemGPT
MemGPT needs the OPENAI_API_KEY environment variable to be set and have a value. It doesn't matter what the value is, it just can't be blank.
```bash
export OPENAI_API_KEY=key-to-success
```
Next we need to configure MemGPT in a way that it will work specifically with sending requests through LiteLLM to Ollama, and process responses back properly. In order to do this we need to make sure that the LLM inference provider is local, the LLM backend is webui, and that we are pointing the default endpoint to our LitelLLM endpoint at http://0.0.0.0:8000. If everything has been done correctly, after finishing the configuration process we should see Saving config to the memgpt config path in terminal.
```bash
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
Now that we have MemGPT running and sending requests to the LiteLLM endpoint we need to test to make sure it's working properly. We'll go through and have a conversation with a MemGPT LLM agent being served from MemGPT through LiteLLM to Ollama
```bash
memgpt run --model openhermes2.5-mistral --model-endpoint http://0.0.0.0:8000 --debug
```

## Install AutoGen
```bash
# Create our the directory for our AutoGen python environment
mkdir -p $HOME/LLM/jttw/autogen/autogen_venv
cd $HOME/LLM/jttw/autogen/

wget -P $HOME/LLM/jttw/autogen/ https://github.com/microsoft/autogen/archive/refs/heads/master.zip
unzip $HOME/LLM/jttw/autogen/master.zip -d $HOME/LLM/jttw/autogen/
rm $HOME/LLM/jttw/autogen/master.zip
cp -r $HOME/LLM/jttw/autogen/autogen-main/. $HOME/LLM/jttw/autogen/
rm -r $HOME/LLM/jttw/autogen/autogen-main

# Create our AutoGen python environment
python3 -m venv $HOME/LLM/jttw/autogen/autogen_venv
# Activate into it, install/update pip, clear python package cache
source $HOME/LLM/jttw/autogen/autogen_venv/bin/activate
python3 -m pip install --upgrade pip
pip3 cache purge
# Install AutoGen python package
pip3 install pyautogen --upgrade
```
## Test AutoGen
Now that we have AutoGen in place we need to test to make sure it's working properly. We'll create a python script in our $HOME/LLM/jttw/autogen/ directory and name the file autogen_litellm_test.py.
```python
from autogen import UserProxyAgent, ConversableAgent

# Create our configuration for the LiteLLM endpoint. API Key is required but the value can be anything.
config_list = [
    {
        "model": "ollama/openhermes2.5-mistral",
        "base_url": "http://0.0.0.0:8000",
        "api_key": "key-to-success"
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
```
Next we'll enter our python environment for AutoGen and run the python test script. If everything is working properly we should be having a conversation with a Autogen LLM agent being served from Autogen through LiteLLM to Ollama.
```bash
cd $HOME/LLM/jttw/autogen/
source $HOME/LLM/jttw/autogen/autogen_venv/bin/activate
python3 autogen_litellm_test.py
```
## Preparing JTTW
```bash
mkdir -p $HOME/LLM/jttw/jttw_venv
cd $HOME/LLM/jttw/
python3 -m venv $HOME/LLM/jttw/jttw_venv
source $HOME/LLM/jttw/jttw_venv/bin/activate
pip3 install guidance
pip3 install litellm==1.14.1
pip3 install pyautogen
pip3 install pymemgpt
```

