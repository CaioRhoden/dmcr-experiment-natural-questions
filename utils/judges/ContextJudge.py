from utils.judges import BaseJudge
from dmcr.models import GenericVLLMBatch
import re


class ContextJudge(BaseJudge):

    def init(self, prompt:str, model: GenericVLLMBatch, model_configs: dict):
        self.prompt = prompt
        self.model = model
        self.model_configs = model_configs

    def judge(self, generations: list[str], questions: list[str], contexts: list[str], regex_pattern: str) -> list[list[float]]:
        prompts = [self._format_input(q, g, c) for q, g, c in zip(questions, generations, contexts)]

        _results = self.model.run(prompts, config_params=self.model_configs)

        ## Iterate over a list of dicts with "generated_text" as key and extract the score using the regex pattern
        scores = []
        for res in _results:
            _fromatted_generations = []
            
            for gen in res:
                match = re.search(regex_pattern, gen['generated_text'])

                if match:
                    _m = match.group(1)
                    try:
                        _fromatted_generations.append(int(_m) if (int(_m) == 1 or int(_m) == 0) else None)
                    except ValueError:
                        _fromatted_generations.append(None)

                else:
                    _fromatted_generations.append(None)

            scores.append(_fromatted_generations)

        return scores
    
    def _format_input(self, question: str, answer: str, context: str) -> str:
        return f""""
                    [System] 
                    {self.prompt}
                    [Question] 
                    {question}
                    [Context]
                    {context}
                    [The Start of Assistant’s Answer] 
                    {answer}
                    [The End of Assistant’s Answer]
                """