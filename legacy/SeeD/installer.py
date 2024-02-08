from pyheaven import *

from transformers import AutoConfig, AutoTokenizer, AutoModel

from seed import *
    
if __name__=="__main__":
    # Initialize config
    api_key = input("Enter your OpenAI API key: ")
    organization = input("Enter your OpenAI organization ID: ")
    clr_config()
    set_config("openai-api", {
        "api_key": api_key,
        "organization": organization,
    })
    set_config("projects_path", "./projects/")
    
    # Localize frozen model
    # Attempted:
    # identifier = "bert-large-uncased" # 1024
    # identifier = "bert-large-vased" # 1024
    identifier = "sentence-transformers/all-MiniLM-L12-v2" # 384
    # identifier = "princeton-nlp/sup-simcse-bert-large-uncased" # 1024
    # ...
    
    # identifier = "princeton-nlp/sup-simcse-bert-large-uncased" # 1024
    C = AutoConfig.from_pretrained(identifier);     C.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    T = AutoTokenizer.from_pretrained(identifier);  T.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    M = AutoModel.from_pretrained(identifier);      M.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    set_config("ckpt-frozen", pjoin(FROZEN_CKPTS_PATH, identifier))
    set_config("cache-dim", 384)
    
    identifier = "bert-large-cased"
    C = AutoConfig.from_pretrained(identifier);     C.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    T = AutoTokenizer.from_pretrained(identifier);  T.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    M = AutoModel.from_pretrained(identifier);      M.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    set_config("ckpt-clslm", pjoin(FROZEN_CKPTS_PATH, identifier))
    
    identifier = "t5-large"
    C = AutoConfig.from_pretrained(identifier);     C.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    T = AutoTokenizer.from_pretrained(identifier);  T.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    M = AutoModel.from_pretrained(identifier);      M.save_pretrained(pjoin(FROZEN_CKPTS_PATH, identifier))
    set_config("ckpt-seqlm", pjoin(FROZEN_CKPTS_PATH, identifier))