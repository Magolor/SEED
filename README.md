# SEED: Domain-Specific Data Curation With Large Language Models

This is the code repo for the paper "SEED: Domain-Specific Data Curation With Large Language Models".

SEED is an approach that leverages Large Language Models (LLMs) to automatically generate domain-specific data curation solutions. By describing a task, input data, and expected output, the SEED compiler produces an executable pipeline consisting of LLM-generated code, small models, and data access modules.

# SEED: Legay Version

[Link to the Legacy Version](https://github.com/Magolor/SEED/tree/main/legacy/SeeD)

# SEED: Dev Version

## Info

**Current Version**: v0.2.0

**Compatibility**: Tested on MacBook M1 Pro, MacOS 12.6.2, Python 3.9, Pytorch 1.13.1

**Hardware Requirements**:  None

## Installation

First install the prerequisites and the SEED package.

```bash
git clone git@github.com:Magolor/SEED.git
cd ./SEED/SeeD/
pip install -r requirements.txt
pip install -e .
cd ..
```

The most basic config is the auth key to OpenAI API. You can create a `openai_api.json` file to store it:
```json
{
    "model": "gpt-4-turbo-preview",
    "api_key": "<YOUR_API_KEY>",
    "organization": "<YOUR_ORGANIZATION>"
}
```
Or you will be asked to manually input them in terminals during SEED setup.

Then run the installer to initialize configurations:
```python
python post_install.py
```

## Tutorials

1. **Recommended**: A full tutorial for understanding how SEED works in general: [amazon_google_full_tutorial](https://github.com/Magolor/SEED/tree/main/tutorials/amazon_google/amazon_google_full_tutorial.ipynb).
2. A short version of the same amazon google project: [amazon_google_tutorial](https://github.com/Magolor/SEED/tree/main/tutorials/amazon_google/amazon_google.ipynb).
3. A code generation agent tutorial: [restaurant_tutorial](https://github.com/Magolor/SEED/tree/main/tutorials/restaurant/restaurant.ipynb).
4. Others:
    - [pubmed_tutorial](https://github.com/Magolor/SEED/tree/main/tutorials/pubmed/pubmed.ipynb)
    - ...

## TODO

SEED is currently under development, many features and optimizations coming!

- [ ] Improve code generation.
- [ ] Add Seq2Seq model support from legacy.
- [ ] Add tools agent from legacy.
- [ ] Improve RAG.
- [x] Improve hyperparaeter search.
- [ ] Integrate hyperparaeter search.
- [ ] Support multiple outputs.
- [ ] ...
