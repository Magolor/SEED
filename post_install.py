from pyheaven import *
from seed import *

from transformers import AutoConfig, AutoTokenizer, AutoModel

if __name__=="__main__":    
    # Initialize config
    if ExistFile("openai_api.json"):
        llm_config = LoadJson("openai_api.json")
    else:
        api_key = input("Enter your OpenAI API key: ")
        organization = input("Enter your OpenAI organization ID: ")
        llm_config = {
            "api_key": api_key,
            "organization": organization,
        }
    init_config(llm_config = llm_config)
    # init_exact_cache()
    init_ckpts()