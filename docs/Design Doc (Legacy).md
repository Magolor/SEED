# SEED Design Document

## Goals

(*Italic* for advanced features)

- [ ] LLM query
  - [ ] Exact cache
  - [ ] Keeping all chat history
  - [ ] *Retrieval-Augmented Generation (RAG)*
- [ ] Code generation
  - [ ] Advisor in code generation
  - [ ] Verfication in code generation
  - [ ] Advisor in code fixing
  - [ ] Code filter
  - [ ] Code ensemble
  - [ ] *Keeping all code versions*
  - [ ] *Manual intervention*
- [ ] Model generation
  - [ ] Custom model
  - [ ] Data accumulation
  - [ ] Dynamic update
  - [ ] Distillation
  - [ ] *Automatic checkpoint selection*
  - [ ] *Support local LLM (7B)*
- [ ] Query batching
  - [ ] Asynchronous batching
- [ ] Iterative tools invocation
  - [ ] *Generated code serves as a tool*
  - [ ] *Generated model serves as a tool*
  - [ ] *Advisor in tools invocation: interaction template*
  - [ ] *Tools invocation result as advice to aid code generation*
  - [ ] *Tools proposal*
- [ ] *Profile: ratio of each local agent, execution time*
- [ ] Generalizability for new templates
- [ ] *Generalizability for new local agents*
- [ ] *Meta-Manager: decide what local agents to use*
- [ ] *New GPT-4 API: function call & assistant*
- [ ] *CoT anywhere*

## Workflow

Project structure:
```
.
└── /<PROJECT_NAME>/
    ├── __init__.py
    ├── configs/
    |   ├── config.json
    |   ├── agent_<X>_config.json
    |   └── ...
    ├── prompt_templates/
    |   └── ...
    ├── code_templates/
    |   └── ...
    ├── apis/
    |   ├── __init__.py
    |   ├── <TOOL_1>/
    |   |   ├── __init__.py
    |   |   └── ...
    |   ├── agent_<X>/
    |   |   ├── __init__.py
    |   |   └── ...
    |   └── ...
    ├── data/
    |   ├── samples.jsonl
    |   ├── test.jsonl
    |   └── ...
    ├── models/
    ├── workspace/
    ├── logs/
    └── ...
```

Each task is associated with a `json` config file. The `name`, `task_desc`, `inputs`, `outputs` should be provided by the user.
```json
{
  "name": ...,
  "task_desc": ...,
  "inputs": [
    {
      "name": ...,
      "type": ...,
      "desc": ...,
    },
    ...,
  ],
  "outputs": [
    {
      "name": ...,
      "type": ...,
      "desc": ...,
    },
    ...,
  ],
  "preset": {
    "instance_prompt": <PATH>,
    "example_prompt": <PATH>,
    "single_query_code": <PATH>,
    ...,
  },
  "agent_<X>_activate": true,
  "agent_<X>_config": <PATH>,
  ...,
}
```

During compilation, every component (tools, local agents, models) are compiled into one API using the config file. The APIs are defined in `/<PROJECT_NAME>/apis/<API_NAME>/__init__.py` of each folder, and the main API provided to the user is defined in the `/<PROJECT_NAME>/__init__.py`.

When running, each instance is associated with a workspace:
```json
{
  "inputs": {
    ...,
  },
  "outputs": {
    ...,
  },
  "agent_<X>_response": {
    "abstain": ...,
    ...,
  },
  ...,
}
```

Each agent updates the workspace by accepting `inputs` and update intermediate responses. Finally the routing program generate `outputs` using the intermediate responses.

During inference: Running the API for each data record
1. Upon calling the API, generate one instance (`json` file) in the local workspace
2. Upon flushing the API, examine and run on all instances in the workspace, and collect the output

## System Utils Implementation

- [ ] File operations
- [ ] Config load & save
- [ ] Workspace load & save?
- [ ] Custom string formatting (to avoid conflict with escape characters)
- [ ] Dynamic Python value eval (to align LLM outputs with the system)
- [ ] `LLMQuery(messages) -> response`
- [ ] `LLMTools(messages, tools) -> response`
- [ ] Response-to-code parsing (response -> code string)
- [ ] Code-to-api parsing? (code string -> code api)
- [ ] Automatic Model load & training (to align different models in huggingface)
- [ ] Sandbox with exception handle

## Code Generation

Advice generation: config + samples -> code advice
Code generation: config + samples + code advice -> code string -> api
Code verificatrion: config + samples + api -> errors
Code evaluation: config + samples + api -> profile
Code fixing advice generation: config + code string + errors -> code advice
Code fixing: config + code string + errors + code advice -> code string -> api
Code branching: config + code string + ???? -> code string -> api
Code ensembling: config + apis -> api

## Model Generation

Advice generation: config + samples -> initial checkpoint
Model generation: initial checkpoint -> models/checkpoint
Model updating: workspace instance -> models/collected_data.jsonl (trigger training -> models/checkpoint)

## LLM Query

Single query: config -> api
Single query infernece: api + samples + workspace instance -> workspace instance
Batch query: config -> api
Batch query infernece: api + samples + workspace instances -> workspace instances
Tools query: config + apis -> api
Tools query infernece: api + samples + workspace instance -> workspace instance
Advised tools query: config + samples + tools -> advice
Advised tools query inference: api + samples + workspace instance + advice -> workspace instance
Tools generation: config -> tools desc + samples

## Routing

Generation: config + apis -> api
Inference: samples -> workspace instances -> workspace instances; agent response -> output

## Profile Generation

workspace instance (agent response) -> profile

## User Perspective

1. Create new project with guide (generating a default project folder)
2. Edit configuration
3. Setup data
4. Compile
5. Inference
6. Adjust configuration based on profile, and re-compile