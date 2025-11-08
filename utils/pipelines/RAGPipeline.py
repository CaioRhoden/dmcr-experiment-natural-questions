
from ast import parse
from html import parser
from tracemalloc import start
from typing import Optional, Literal
from unittest.mock import Base
import polars as pl
import os
import torch
import datetime
import faiss
import json
from FlagEmbedding import FlagModel
import wandb

from dmcr.models import  BaseLLM, BatchModel, GenericVLLMBatch
BATCH_MODEL_LM = Literal["hf", "vllm"]

from utils.set_random_seed import set_random_seed
class RAGPipeline:
    def __init__(
        self,
        # We define a parameter for each field in the Config dataclass
        retrieval_path: str,
        wiki_path: str,
        embeder_path: str,
        vector_db_path: str,
        questions_path: str,
        project_log: str,
        k: int,
        size_index: int,
        instruction: str,
        tags: list[str] = [],
        model_run_id: str = "rag_generations",
        lm_configs: Optional[dict[str, float|int]] = None,
        lm_instance_kwargs: dict = {},
        root_path: str = ".",
        seed: Optional[int] = None,
        attn_implementation: str = "sdpa",
        thinking: bool = False,
        batch_size: int = 1,
        log: bool = False,
        language_model_path: str = "",
        **kwargs, # Use kwargs to gracefully handle any extra fields
    ):
        
        '''
        Initializes the experiment pipeline with individual configuration arguments.
        '''
        # Assign all initialization parameters to instance attributes

        self.retrieval_path = retrieval_path
        self.wiki_path = wiki_path
        self.embeder_path = embeder_path
        self.vector_db_path = vector_db_path
        self.questions_path = questions_path
        self.project_log = project_log
        self.model_run_id = model_run_id
        self.k = k
        self.size_index = size_index
        self.log = log

        ## Pre collection parameters
        self.instruction = instruction
        self.lm_configs = lm_configs if lm_configs is not None else {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
        }
        self.lm_instance_kwargs = lm_instance_kwargs
        self.tags = tags
        self.log = True
        self.root_path = root_path
        self.seed = seed
        self.batch_size = batch_size
        self.attn_implementation: str = attn_implementation
        self.thinking = thinking
        self.language_model_path = language_model_path

        if self.seed is not None:
            set_random_seed(self.seed)


    def setup(self):

        """
        Setup for the experiment.

        This function downloads the 100 question golden dataset, writes it to questions.feather and creates the retrieval, generations, results and datamodels folders.
        """
        ## Create structure
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.exists(f"{self.root_path}/retrieval"):
            os.mkdir(f"{self.root_path}/retrieval")
        if not os.path.exists(f"{self.root_path}/generations"):
            os.mkdir(f"{self.root_path}/generations")
        
        if not os.path.exists(f"{self.root_path}/logs"):
            os.mkdir(f"{self.root_path}/logs")

    def _parse_generation_output(self, output: dict) -> str:
        """
        Parse the output of the generation model, analyze if is it "enable_thinking"

        Parameters:
        - output (str): The raw output from the generation model.

        Returns:
        - str: The parsed output.
        """
        
        parsed_output = []
        for out in output:

            if self.thinking:
                # Example parsing logic for "enable_thinking"
                # This is a placeholder; replace with actual logic as needed
                parsed_output.append(str(out["generated_text"].split("</think>")[-1].strip()))
            else:
                parsed_output.append(str(out["generated_text"]))
        
        return parsed_output

    def get_rag_retrieval(self):

        ## Setup variables
        """
        Load the faiss indices and iterate questions to get the l2 and ip retrieval data for each question.
        This function writes the retrieval data into retrieval_data.json in the retrieval folder.

        Parameters:
        None

        Returns:
        None
        """
        


        retrieval_indexes = {}
        retrieval_distances = {}

        df = pl.read_ipc(self.questions_path)

        ### Load faiss indices
        index = faiss.read_index(self.vector_db_path)
        # ip_index = faiss.read_index(IP_FAISS_INDEX_PATH)
        embedder = FlagModel(self.embeder_path, devices=["cuda:0"], use_fp16=True, batch_size=1)
        
        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"RAG_retrieval_{self.model_run_id}",
                id = f"RAG_retrieval_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "index": self.vector_db_path,
                    "questions_path": self.questions_path,
                    "embeder_path": self.embeder_path,

                },
                tags = self.tags.extend(["RAG", "retrieval"]),
            )
            
            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})





        ### Iterate questions
        for idx in range(len(df)):

            question = df[idx]["question"].to_numpy().flatten()[0]
            print(f"Question {idx}: {question}")
            query_embedding = embedder.encode(
                question,
                convert_to_numpy=True,
            )
            query_embedding = query_embedding.astype('float32').reshape(1, -1)

            ### Get l2 and ip neighbors
            scores, ids = index.search(query_embedding, self.size_index)
            # ip_ids, ip_scores = ip_index.search(query_embedding, 100)

            retrieval_indexes[idx] = ids.tolist()[0]
            retrieval_distances[idx] = scores.tolist()[0]
            # retrieval_data["ip"][idx] = (ip_ids.tolist()[0], ip_scores.tolist()[0])

        ## Save into json
        with open(f"{self.root_path}/retrieval/rag_retrieval_indexes.json", "w") as f:
            json.dump(retrieval_indexes, f)

        with open(f"{self.root_path}/retrieval/rag_retrieval_distances.json", "w") as f:
            json.dump(retrieval_distances, f)

        if self.log:
            wandb.log({
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "duration": (datetime.datetime.now() - start_time).total_seconds()
            })
            artifact = wandb.Artifact(name="rag_retrieval", type="json", description="RAG retrieval data")
            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_indexes.json")
            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_distances.json")
            wandb.log_artifact(artifact)
            wandb.finish()

    def get_rag_generations(self, model: None | BatchModel | BaseLLM = None):
        """
        Get RAG generations for the given batch model language model.

        Args:
            batch_model_lm (BATCH_MODEL_LM): The batch model language model to use.

        Returns:
            None
        """

        wiki = pl.read_ipc(self.wiki_path).with_row_index("idx")
        questions = pl.read_ipc(self.questions_path)
        ## Load retrieval data
        with open(f"{self.root_path}/retrieval/rag_retrieval_indexes.json", "r") as f:
            retrieval_data = json.load(f)

        batch_list = []

        if model is None:
            print("Loading default model...")
            model = GenericVLLMBatch(
                path=self.language_model_path,
                thinking=False,
                vllm_kwargs={"max_model_len": 32768}
            )

        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"RAG_generations_{self.model_run_id}",
                id = f"RAG_generations_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "index": self.vector_db_path,
                    "questions_path": self.questions_path,
                    "embeder_path": self.embeder_path,

                },
                tags = self.tags.extend(["RAG", "generations"]),
            )
            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})

        generations = {}
            

        ## Iterate questions
        for idx in range(len(retrieval_data)):

            top_k = retrieval_data[f"{idx}"][0:self.k]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                prompt += f"Document[{self.k-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['text'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[idx]['question'].to_numpy().flatten()[0]}\nAnswer: "
            
            if self.batch_size > 1:
                if len(batch_list) < self.batch_size:
                    batch_list.append((idx, prompt))
                
                if len(batch_list) == self.batch_size or idx == (len(questions) - 1):

                    outputs = model.run(
                        [str(_q[1]) for _q in batch_list], 
                        instruction=self.instruction, 
                        config_params=self.lm_configs
                    )
                    for i, _q in enumerate(batch_list):
                        generations[f"{_q[0]}"] = self._parse_generation_output(outputs[i])
                    batch_list = []
                
                    if self.model_run_id is None:
                        path = f"{self.root_path}/generations/rag_generations.json"
                        with open(path, "w") as f:
                            json.dump(generations, f)
                    
                    else:
                        path = f"{self.root_path}/generations/{self.model_run_id}.json"
                        with open(path, "w") as f:
                            json.dump(generations, f)

            else:
                ## Generate output
                outputs = model.run(
                    prompt, 
                    instruction=self.instruction, 
                    config_params=self.lm_configs
                )

                generations[f"{idx}"] = self._parse_generation_output(outputs)

                if self.model_run_id is None:
                    path = f"{self.root_path}/generations/rag_generations.json"
                    with open(path, "w") as f:
                        json.dump(generations, f)
                
                else:
                    path = f"{self.root_path}/generations/{self.model_run_id}.json"
                    with open(path, "w") as f:
                        json.dump(generations, f)
            
        if self.log:
            wandb.log({
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "duration": (datetime.datetime.now() - start_time).total_seconds()
            })
            artifact = wandb.Artifact(name=f"{self.model_run_id}", type="json", description="RAG generations")
            artifact.add_file(path)
            wandb.log_artifact(artifact)

            wandb.finish()

        ## Save into json
        
        

