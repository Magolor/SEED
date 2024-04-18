
from seed import *
from .topic_classification_singlequery import topic_classification_single_class
from .topic_classification_batchquery import topic_classification_batch_class
from .topic_classification_cache import topic_classification_cache_class
from .topic_classification_simul import topic_classification_simul_class
from . import topic_classification_ensembled

class topic_classification_integrated_class(object):
    def __init__(self, args):
        self.use_cache = args.use_cache
        self.use_codegen = args.use_codegen
        self.use_simul = args.use_simul
        self.use_batch = args.use_batch
        self.no_query = args.no_query
        if self.use_cache:
            self.cache_service = topic_classification_cache_class(module_path=pjoin("modules", "topic_classification"), cache_threshold=args.cache_threshold)
        if self.use_simul:
            self.simul_service = topic_classification_simul_class(module_path=pjoin("modules", "topic_classification"), simul_type=args.simul_type, simul_threshold=args.simul_threshold, num_classes=args.simul_num_classes)
        if self.use_batch:
            self.service = topic_classification_batch_class(batch_size=args.batch_size)
        else:
            self.service = topic_classification_single_class()
        self.clear()
    
    def clear(self):
        self.inputs = list()
        self.indices = list()
        self.responses = list()
        self.annotated = list()
        self.progress = 0
        if self.use_cache:
            self.cache_service.clear()
        if self.use_simul:
            self.simul_service.clear()
        self.service.clear()
    
    def update(self):
        if len(self.indices)!=0:
            return
        if self.use_cache:
            for data_input, response, annotated in zip(self.inputs[self.progress:], self.responses[self.progress:], self.annotated[self.progress:]):
                if annotated and (response is not None):
                    self.cache_service.update(value=response, **data_input)
        if self.use_simul:
            for data_input, response, annotated in zip(self.inputs[self.progress:], self.responses[self.progress:], self.annotated[self.progress:]):
                if annotated and (response is not None):
                    self.simul_service.update(value=response, **data_input)
        self.progress = len(self.responses)
        
    def __call__(self, text):
        self.inputs.append(Data(
            text=text
        ))
        response = None
        if self.use_cache:
            self.cache_service(
                text=text
            ); response = self.cache_service.responses.pop()
            if response is not None:
                add_llm_count(type="cache", size=1)
                self.responses.append(response)
                self.annotated.append(False)
                self.update()
                return len(self.indices)==0
        if self.use_codegen:
            response = topic_classification_ensembled.topic_classification(
                text=text
            )
            if response is not None:
                add_llm_count(type="codegen", size=1)
                self.responses.append(response)
                self.annotated.append(False)
                self.update()
                return len(self.indices)==0
        if self.use_simul:
            self.simul_service(
                text=text
            ); response = self.simul_service.responses.pop()
            if response is not None:
                add_llm_count(type="simul", size=1)
                self.responses.append(response)
                self.annotated.append(False)
                self.update()
                return len(self.indices)==0

        self.indices.append(len(self.responses))
        self.responses.append(None)
        self.annotated.append(False)
        if not self.no_query:
            flushed = self.service(
                text=text
            )
            if flushed:
                for idx, response in zip(self.indices, self.service.responses):
                    self.responses[idx] = response; self.annotated[idx] = True
                self.service.clear()
                self.indices = list()
                self.update()
                return True
            return False
        else:
            self.service.clear()
            self.indices = list()
            self.update()
            return True
    
    def flush(self):
        if self.use_batch:
            self.service.flush()
            for idx, response in zip(self.indices, self.service.responses):
                self.responses[idx] = response; self.annotated[idx] = True
            self.service.clear()
            self.indices = list()
            self.update()
