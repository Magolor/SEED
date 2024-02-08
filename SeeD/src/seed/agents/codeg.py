from ..templates import *
from .agent import Agent

class CodeGenAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = pjoin(self.config['project_path'], "agents", "codeg")
        
    def compile(self, codeg_examples=list(), codev_examples=list()):
        ClearFolder(self.path, rm=True); CreateFile(pjoin(self.path,'__init__.py'))
        
        llm = LLM()
        api = API(**self.config)
        SaveJson(codeg_examples, pjoin(self.path, "codeg_examples.jsonl"), backend="jsonl")
        SaveJson(codeg_examples+codev_examples, pjoin(self.path, "natural_examples.jsonl"), backend="jsonl")
        # TODO: examples augmentation
        
        # Advices Generation
        advices_prompt = format_advsg_prompt(**self.config)
        response = llm.q(messages = [
            {'role': 'user', 'content': advices_prompt},
        ], post_processings=[partial(parse_ideas, limit=self.config['codeg_branches_count'])])
        ideas = [response[idea] for idea in [f"idea {i}" for i in range(1, self.config['codeg_branches_count']+1)] if response[idea] is not None]
        
        validation_examples = LoadJson(pjoin(self.path, "natural_examples.jsonl"), backend="jsonl")
        
        # Branches Generation
        branches = list()
        for sid, idea in enumerate(ideas):
            codeg_prompt = format_codeg_prompt(advice=idea, examples=codeg_examples, **self.config)
            response = llm.q(messages = [
                {'role': 'user', 'content': codeg_prompt},
            ], post_processings = [parse_code])
            code = response['code'] if response['code'] else None
            if code is not None:
                profile = self.codeg_evalute_examples(code, validation_examples)
                branches.append((sid, "0", code, idea, [codeg_prompt], profile))
        for sid, vid, code, idea, history, profile in branches:
            self.save_snippet(sid, vid, code, idea, history, profile)
                
        # Evolution
        for T in range(self.config['codeg_iterations']):
            # Branching
            new_branches = []
            for sid, vid, code, idea, history, profile in branches:
                new_branches.append((sid, vid, code, idea, history, profile))
                failures = profile['failures']
                if len(failures) > self.config['codeg_branching']:
                    failures = [failures[i] for i in np.random.choice(len(failures), self.config['codeg_branching'], replace=False)]
                for i, failure in enumerate(failures):
                    example, result = failure
                    if (not result['status']):
                        example['error'] = result['msg']
                    else:
                        example['responses'] = {api.output: result['response']}
                    fixing_prompt = format_codeg_expfix_prompt(
                        code = comment(idea) + '\n\n' + code,
                        example = example,
                        **self.config
                    )
                    response = llm.q(messages = [
                        {'role': 'user', 'content': fixing_prompt},
                    ], post_processings = [parse_code])
                    code = response['code'] if response['code'] else None
                    if code is not None:
                        profile = self.codeg_evalute_examples(code, validation_examples)
                        new_branches.append((sid, vid+f"{i}", code, idea, history+[fixing_prompt], profile))
            for sid, vid, code, idea, history, profile in new_branches:
                self.save_snippet(sid, vid, code, idea, history, profile)
            # Filtering
            # 1. Dominance
            dominated = []
            for branch1_id, branch2_id in combinations(range(len(new_branches)), 2):
                if self.codeg_profile_dominated(new_branches[branch1_id][-1], new_branches[branch2_id][-1]):
                    dominated.append(branch2_id)
                elif self.codeg_profile_dominated(new_branches[branch2_id][-1], new_branches[branch1_id][-1]):
                    dominated.append(branch1_id)
            undominated = [i for i in range(len(new_branches)) if i not in dominated]
            new_branches = [new_branches[i] for i in undominated]
            # 2. Sorting
            def filter_metric(profile):
                return (profile[self.config['evaluation_metric']], profile['correct_ratio'], -profile['failure_ratio'])
            new_branches = sorted(new_branches, key=lambda x: filter_metric(x[-1]), reverse=True)
            branches = new_branches[:self.config['codeg_branches_count']]
        
        list_of_snippets = [f"snippet_s{sid:03d}_v{vid}" for sid, vid, code, idea, history, profile in branches]
        with open(pjoin(self.path,'__init__.py'), 'w') as f:
            f.write(format_ensemble_code(list_of_snippets=list_of_snippets, **self.config))
        

    def codeg_profile_dominated(self, profile1, profile2):
        # If profile1 has a lower metric value, it is definitely not dominating profile2
        if profile1[self.config['evaluation_metric']] < profile2[self.config['evaluation_metric']]:
            return False
        # If profile1 has a lower correct_ratio, it is defnitely not dominating profile2
        if profile1['correct_ratio'] < profile2['correct_ratio']:
            return False
        # If profile1 has a higher failure_ratio, it is definitely not dominating profile2
        if profile1['failure_ratio'] > profile2['failure_ratio']:
            return False
        # If profile2 has correct examples that profile1 does not have, profile1 is not dominating profile2
        for correct in profile2['corrects']:
            if correct not in profile1['corrects']:
                return False
        # If profile1 has failures that profile2 does not have, profile1 is not dominating profile2
        for failure in profile1['failures']:
            if failure not in profile2['failures']:
                return False
        # Now that profile2 is no better than profile1, if profile1 has more correct examples, it is dominating profile2
        # If profile1 has correct examples that profile2 does not have, profile1 is dominating profile2
        for correct in profile1['corrects']:
            if correct not in profile2['corrects']:
                return True
        # If profile2 has failures that profile1 does not have, profile1 is dominating profile2
        for failure in profile2['failures']:
            if failure not in profile1['failures']:
                return True
        # Now that profile2 is no better but no worse than profile1, profile1 is not dominating profile2
        return False

    def codeg_evalute_examples(self, code, examples):
        api = API(**self.config)
        metric = get_evaluation_metric(self.config['evaluation_metric'])
        corrects, failures, abstains = list(), list(), list()
        pds, gts = list(), list()
        for example in examples:
            gt = example['outputs'][api.output]
            result = api.api_execute(code, example['inputs'])
            if (not result['status']):
                pds.append(None); gts.append(gt)
                failures.append((example, result))
            else:
                pd = result['response']
                pds.append(pd); gts.append(gt)
                if (pd is None):
                    abstains.append((example, result))
                elif metric.compute(references=[gt], predictions=[pd])[self.config['evaluation_metric']] < 1.0:
                    failures.append((example, result))
                else:
                    corrects.append((example, result))
        non_abs_gt_value = [gt for (gt, pd) in zip(gts, pds) if pd is not None]
        non_abs_pd_value = [pd for (gt, pd) in zip(gts, pds) if pd is not None]
        return {
            'corrects': corrects,
            'failures': failures,
            'abstains': abstains,
            'correct_ratio': (len(gts)-len(failures)-len(abstains))/len(gts),
            'failure_ratio': len(failures)/len(gts),
            'abstain_ratio': len(abstains)/len(gts),
        } | metric.compute(references=non_abs_gt_value, predictions=non_abs_pd_value)
    
    def save_snippet(self, sid, vid, code, idea, history, profile):
        doced_code = "\n\n".join(
            [comment(f"Snippet ID: {sid}\nVersion ID: {vid}"), comment(idea)]
        +   [comment(h) for h in history]
        +   [code]
        +   [comment("\n".join([f"{k}: {v}" for k, v in profile.items() if k not in ['corrects', 'failures', 'abstains']]))]
        )
        with open(pjoin(self.path,f'snippet_s{sid:03d}_v{vid}.py'), 'w') as f:
            f.write(doced_code)