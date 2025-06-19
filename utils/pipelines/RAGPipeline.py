
from typing import Optional
import polars as pl
import os
import torch
import datetime
import faiss
import json
from FlagEmbedding import FlagModel
import wandb



from dmcr.datamodels.config import LogConfig
from dmcr.models import GenericInstructModelHF


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
        laguage_model_path: str,
        project_log: str,
        model_run_id: str,
        k: int,
        size_index: int,
        instruction: str,
        tags: list[str] = [],
        lm_configs: Optional[dict[str, float]] = None,
        root_path: str = ".",
        seed: int|None = None,
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
        self.laguage_model_path = laguage_model_path
        self.project_log = project_log
        self.model_run_id = model_run_id
        self.k = k
        self.size_index = size_index

        ## Pre collection parameters
        self.instruction = instruction
        self.lm_configs = lm_configs if lm_configs is not None else {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048.0,
            "max_new_tokens": 10.0,
            "num_return_sequences": 5.0
        }
        self.tags = tags
        self.log = True
        self.root_path = root_path
        self.seed = seed

        if self.seed is not None:
            set_random_seed(self.seed)


    def setup(self):

        """
        Setup for the experiment.

        This function downloads the 100 question golden dataset, writes it to questions.feather and creates the retrieval, generations, results and datamodels folders.
        """
        ## Create structure
        os.mkdir(f"{self.root_path}/retrieval")
        os.mkdir(f"{self.root_path}/generations")

        os.mkdir(f"{self.root_path}/logs")


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
        embedder = FlagModel(self.embeder_path, devices=["cuda:0"], use_fp16=True)
        
        if self.log:
            wandb.init(
                project=self.project_log,
                name=f"RAG_retrieval_{self.model_run_id}",
                id = f"RAG_{self.model_run_id}_{str(datetime.datetime.now())}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": self.vector_db_path,
                    "questions_path": self.questions_path,
                    "embeder_path": self.embeder_path,

                },
                tags = self.tags.extend(["RAG", "retrieval"]),
            )

            wandb.log({"start_time": str(datetime.datetime.now())})





        ### Iterate questions
        for idx in range(len(df)):

            question = df[idx]["question"].to_numpy().flatten()[0]
            query_embedding = embedder.encode(
                [question],
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
            artifact = wandb.Artifact(name="rag_retrieval", type="json", description="RAG retrieval data")
            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_indexes.json")
            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_distances.json")
            wandb.log_artifact(artifact)
            wandb.log({
                "end_time": str(datetime.datetime.now()),
                "duration": str(datetime.datetime.now() - datetime.datetime.strptime(wandb.config["start_time"], "%Y-%m-%d %H:%M:%S.%f"))
            })
            wandb.finish()

    def get_rag_generations(self):


        ## Setup variables
        wiki = pl.read_ipc(self.wiki_path).with_row_index("idx")
        questions = pl.read_ipc(self.questions_path)

        model = GenericInstructModelHF(self.laguage_model_path)


        generations = {}

        ## Load retrieval data
        with open(self.retrieval_path, "r") as f:
            retrieval_data = json.load(f)

        print(retrieval_data)
        ## Iterate questions
        for r_idx in range(len(retrieval_data)):

            top_k = retrieval_data[f"{r_idx}"][0:self.k]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                prompt += f"Document[{self.k-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents ", 
                config_params=self.lm_configs
            )

            generations[f"{r_idx}"] = [str(out["generated_text"]) for out in outputs]

            with open(f"{self.root_path}/generations/rag_generations.json", "w") as f:
                json.dump(generations, f)

        ## Save into json
        
        
    def invoke_pipeline_step(self, step: str):

        """
        This function is used to invoke the pipeline for a specific step.
        
        It uses a match-case statement to determine which step to run.
        
        The steps are:
        - setup: Sets up the experiment.
        - get_rag_retrieval: Get the retrieval data from the RAG model.
        - get_rag_generations: Get the generations data from the RAG model.
        - get_datamodels_generations: Get the generations data from the datamodels.
        - create_datamodels_datasets: Create the datasets for the datamodels.
        - run_pre_collections: Run the pre-collections for the datamodels.
        - run_collections: Run the collections for the datamodels.
        - train_datamodels: Train the datamodels.
        - evaluate_datamodels: Evaluate the datamodels.
        - get_datampodels_retrieval: Get the retrieval data from the datamodels.
        
        The function does not return anything.
        """
        match step:
            case "setup":
                self.setup()

            case "get_rag_retrieval":
                self.get_rag_retrieval()

            case "get_rag_generations":
                self.get_rag_generations()
            
            case _:
                raise ValueError(f"Invalid step: {step}")
        
