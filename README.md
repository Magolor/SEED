# SEED: Domain-Specific Data Curation With Large Language Models

This is the code repo for the paper "SEED: Domain-Specific Data Curation With Large Language Models".

SEED is an approach that leverages Large Language Models (LLMs) to automatically generate domain-specific data curation solutions. By describing a task, input data, and expected output, the SEED compiler produces an executable pipeline consisting of LLM-generated code, small models, and data access modules.

"The code in this repository **is currently undergoing reconstruction** to develop an out-of-the-box solution suitable for real-world applications, which means it could be temporarily unstable in terms of robustness and performance. Our goal is to provide a streamlined and ready-to-use SEED implementation. During this reconstruction phase, we are gradually enhancing the codebase to improve its usability, performance, and compatibility with different use cases. If you need to refer to the old version —— the code used for experiments, please check out the `legacy/` directory.

# Info

**Current Version**: v0.2.0

**Compatibility**: Tested on MacBook M1 Pro, MacOS 12.6.2, Python 3.9, Pytorch 1.13.1

**Hardware Requirements**:  None

# Installation

**Notice that the code is currently undergoing reconstruction and unable to install!**

First install the prerequisites and the SEED package.

```bash
git clone git@github.com:Magolor/SEED.git
cd ./SEED/SeeD/
pip install -r requirements.txt
pip install -e .
cd ..
```

Then run the installer to initialize configurations (including OpenAI keys):
```python
python installer.py
```

# Quickstart

TODO

# Advanced Usage

TODO