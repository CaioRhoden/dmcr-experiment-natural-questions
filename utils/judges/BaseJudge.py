from abc import ABC, abstractmethod
from dmcr.models import GenericVLLMBatch
import polars

class BaseJudge(ABC):

    @abstractmethod
    def init(self, prompt:str, model: GenericVLLMBatch, model_configs: dict, **kwargs):
        '''
        Initialize the judge with the given prompt and model.
        '''
        pass

    @abstractmethod
    def judge(self, generations: list[str], questions: list[str], regex_pattern: str, **kwargs) -> list[float]:
        '''
        Judge the given generations and questions and return a list of scores between 1 and 0.
        '''
        pass