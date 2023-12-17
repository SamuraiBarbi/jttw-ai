# jttw-ai

**Jesus Take the Wheel**, an A.I. assistant for people like me

## Components
- Ollama
- LiteLLM
- Guidance
- MemGPT
- AutoGen
- Aider
- Sweep
- PromptFoo

## Install MemGPT Python Package
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

memgpt run --model dolphin2.2-mistral:7b-q6_K --model-endpoint http://0.0.0.0:8000 --debug
