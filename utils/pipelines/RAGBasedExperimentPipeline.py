
from curses import start_color
from pyexpat import model
from tracemalloc import start
from typing import Optional
import polars as pl
import os
import torch
import datetime
import faiss
import json
from FlagEmbedding import FlagModel
import wandb

from dmcr.datamodels.setter.IndexBasedSetter import IndexBasedSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.models import GenericInstructModelHF
from dmcr.evaluators import Rouge_L_evaluator, SquadV2Evaluator, JudgeEvaluator
from dmcr.datamodels.models import LinearRegressor
from dmcr.datamodels.models import FactoryLinearRegressor


from utils.weights_to_json import load_weights_to_json
from utils.set_random_seed import set_random_seed

class RAGBasedExperimentPipeline:
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
        train_collection_id: str,
        test_collection_id: str,
        k: int,
        size_index: int,
        num_models: int,
        evaluation_metric: str,
        evaluator: str,
        instruction: str,
        
        train_samples: int,
        test_samples: int,
        train_start_idx: int,
        train_end_idx: int,
        test_start_idx: int,
        test_end_idx: int,
        train_checkpoint: int,
        test_checkpoint: int,
        epochs: int,
        lr: float,
        train_batches: int,
        val_batches: int,
        val_size: float,
        patience: int,
        log_epochs: int,
        tags: list[str] = [],
        seed: Optional[int] = None,
        lm_configs: Optional[dict[str, int|float]] = None,
        log: bool = False,
        root_path: str = ".",
        datamodels_generation_name: Optional[str] = "datamodels_generations",
        **kwargs, # Use kwargs to gracefully handle any extra fields
    ):
        
        '''
        Initializes the experiment pipeline with individual configuration arguments.
        '''
        # Assign all initialization parameters to instance attributes
        self.seed = seed

        self.retrieval_path = retrieval_path
        self.wiki_path = wiki_path
        self.embeder_path = embeder_path
        self.vector_db_path = vector_db_path
        self.questions_path = questions_path
        self.laguage_model_path = laguage_model_path
        self.project_log = project_log
        self.model_run_id = model_run_id
        self.train_collection_id = train_collection_id
        self.test_collection_id = test_collection_id
        self.k = k
        self.size_index = size_index
        self.num_models = num_models
        self.evaluation_metric = evaluation_metric
        self.evaluator = evaluator

        ## Pre collection parameters
        self.instruction = instruction
        self.lm_configs = lm_configs if lm_configs is not None else {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 10,
            "num_return_sequences": 5
        }
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.train_start_idx = train_start_idx
        self.train_end_idx = train_end_idx
        self.test_start_idx = test_start_idx
        self.test_end_idx = test_end_idx
        self.train_checkpoint = train_checkpoint
        self.test_checkpoint = test_checkpoint

        ## Training parameters
        self.epochs = epochs
        self.lr = lr
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.val_size = val_size
        self.patience = patience
        self.log_epochs = log_epochs
        self.tags = tags
        self.log = log

        self.root_path = root_path
        self.datamodels_generation_name = datamodels_generation_name
        if seed:
            set_random_seed(seed)



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
        if not os.path.exists(f"{self.root_path}/results"):
            os.mkdir(f"{self.root_path}/results")
        if not os.path.exists(f"{self.root_path}/datamodels"):
            os.mkdir(f"{self.root_path}/datamodels")
        if not os.path.exists(f"{self.root_path}/logs"):
            os.mkdir(f"{self.root_path}/logs")

        ## Create Datamodels Structure
        if not os.path.exists(f"{self.root_path}/datamodels/datasets"):
            os.mkdir(f"{self.root_path}/datamodels/datasets")
        if not os.path.exists(f"{self.root_path}/datamodels/pre_collections"):
            os.mkdir(f"{self.root_path}/datamodels/pre_collections")
        if not os.path.exists(f"{self.root_path}/datamodels/collections"):
            os.mkdir(f"{self.root_path}/datamodels/collections")
        if not os.path.exists(f"{self.root_path}/datamodels/models"):
            os.mkdir(f"{self.root_path}/datamodels/models")


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
            artifact = wandb.Artifact(
                name="rag_retrieval_data",
                type="json",
                description="RAG retrieval daa"
            )

            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_indexes.json")
            artifact.add_file(f"{self.root_path}/retrieval/rag_retrieval_distances.json")
            wandb.log_artifact(artifact)
            end_time = datetime.datetime.now()
            wandb.log({
                "end_time": end_time.strftime('%Y-%m-%d_%H-%M-%S'),
                "total_duration": (end_time - start_time).total_seconds()
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
                instruction= self.instruction, 
                config_params=self.lm_configs
            )

            generations[f"{r_idx}"] = [str(out["generated_text"]) for out in outputs]

            with open(f"{self.root_path}/generations/rag_generations.json", "w") as f:
                json.dump(generations, f)

            if self.log:
                artifact = wandb.Artifact(
                    name="rag_retrieval_data",
                    type="json",
                    description="RAG generation daa"
                )

                artifact.add_file(f"{self.root_path}/generations/rag_generations.json")
                wandb.log_artifact(artifact)
                end_time = datetime.datetime.now()
                wandb.log({
                    "end_time": end_time.strftime('%Y-%m-%d_%H-%M-%S'),
                    "total_duration": (end_time - start_time).total_seconds(),
                })
                wandb.finish()

        ## Save into json
        
            
    def create_datamodels_datasets(self):
        """
        This function creates two .h5 files, training and testing, with respective sizes train_samples and test_samples
        Each element of the dataset corresponds in array of k samples going from [0, size_index)
        These elements represents the position on the RAG dict, as the index for each sample may vary the position in the relative top-size_indez retrieved samples will be
        the same
        """

        DATASET_PATH = f"{self.root_path}/datamodels"
        setter_config = IndexBasedSetterConfig(
            save_path=DATASET_PATH,
            size_index=self.size_index,
            k=self.k,
            train_samples= self.train_samples,
            test_samples= self.test_samples
        )

        setter = IndexBasedSetter(config=setter_config)
        setter.set()
        
    def run_pre_colections(self):


        """
        This function creates the pre collections for train and test datasets.
        It uses the DatamodelsIndexBasedNQPipeline to create the pre collections.
        The function takes the instruction, llm, start_idx, end_idx, mode, log,
        log_config, checkpoint, output_column, model_configs, and rag_indexes_path
        as parameters. It returns nothing.
        """
        model = GenericInstructModelHF(self.laguage_model_path)

        config = DatamodelIndexBasedConfig(
            k = self.k,
            num_models= self.num_models,
            datamodels_path = f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )

        train_log_config = None
        test_log_config = None
        if self.log:
            train_log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"pre_collection_{self.train_collection_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=f"pre_collection_{self.train_collection_id}",
                config={
                    "llm": f"{self.laguage_model_path}",
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "model_configs": self.lm_configs,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend(["train", "pre_collection"])
            )

            test_log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"pre_collection_{self.test_collection_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=f"pre_collection_{self.test_collection_id}",
                config={
                    "llm": f"{self.laguage_model_path}",
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "model_configs": self.lm_configs,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend(["test", "pre_collection"])
                
            )



        datamodel = DatamodelsIndexBasedNQPipeline(config=config)

        print("Start Creating Train Pre Collection")

        datamodel.create_pre_collection(
            instruction= self.instruction,
            llm = model,
            start_idx = self.train_start_idx, 
            end_idx = self.train_end_idx, 
            mode = "train", 
            log = self.log, 
            log_config = train_log_config, 
            checkpoint = self.train_checkpoint, 
            output_column = "answers",
            model_configs = self.lm_configs,
            rag_indexes_path=f"{self.root_path}/retrieval/rag_retrieval_indexes.json"
        )

        print("Start Creating Test Pre Collection")
        datamodel.create_pre_collection(
            instruction= self.instruction,
            llm = model,
            start_idx = self.test_start_idx, 
            end_idx = self.test_end_idx, 
            mode = "test", 
            log = self.log, 
            log_config = test_log_config, 
            checkpoint = self.test_checkpoint, 
            output_column = "answers",
            model_configs = self.lm_configs,
            rag_indexes_path=f"{self.root_path}/retrieval/rag_retrieval_indexes.json"
        )


    def run_collections(self):



        config = DatamodelIndexBasedConfig(
            k = self.k,
            num_models= self.num_models,
            datamodels_path = f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )

        ## Config evaluator based on yaml parameter
        if self.evaluator == "Rouge-L":
            evaluator = Rouge_L_evaluator()
        elif self.evaluator == "SquadV2":
            evaluator = SquadV2Evaluator("best_f1")
        elif self.evaluator == "Judge":
            def format_input(question, response):
                return f""""
                [System] 
                Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a question displayed below. Your evaluation should consider factors such as relevance and accuracy. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".  
                [Question] 
                {question}  
                [The Start of Assistant’s Answer] 
                {response}
                [The End of Assistant’s Answer]
                """
            evaluator = JudgeEvaluator(
                model_path=self.laguage_model_path,
                model_configs = {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_return_sequences": 1,
                },
                instruction="",
                format_template=format_input
            )

        else:
            raise ValueError(f"Invalid evaluator")


        datamodel = DatamodelsIndexBasedNQPipeline(config)
        train_log_config = None
        test_log_config = None
        if self.log:
            test_log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"test_collections_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=self.test_collection_id,
                config={
                    "evaluator": "Rouge-L",
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend(["test", "collections"])
            )

            train_log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"train_collections_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=self.train_collection_id,
                config={
                    "evaluator": self.evaluator,
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend(["train", "collections"])
            )

        print("Start Creating Train Collection")
        datamodel.create_collection(
            evaluator = evaluator,
            mode = "train",
            collection_name = self.train_collection_id,
            log = self.log,
            log_config = train_log_config,
            checkpoint = self.train_checkpoint,
            start_idx = self.train_start_idx,
            end_idx = self.train_end_idx,
        )


        print("Start Creating Test Collection")
        datamodel.create_collection(
            evaluator = evaluator,
            mode = "test",
            collection_name = self.test_collection_id,	
            log = self.log,
            log_config = test_log_config,
            checkpoint = self.test_checkpoint

        )


    def train_datamodels(self):

        epochs = self.epochs
        lr = self.lr
        train_batches = self.train_batches
        val_batches = self.val_batches
        val_size = self.val_size
        patience = self.patience
        log_epochs = self.log_epochs


        config = DatamodelIndexBasedConfig(
            k = self.k,
            num_models= self.num_models,
            datamodels_path = f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )



        datamodel = DatamodelsIndexBasedNQPipeline(config)
        log_config = None
        if self.log:
            log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"test_train_datamoles_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=self.model_run_id,
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config),

                },
                tags=self.tags.extend(["training"])
            )



        model_factory = FactoryLinearRegressor(
            in_features=self.size_index,
            out_features=1,
            device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )

        datamodel.train_datamodels(
            model_factory=model_factory,
            collection_name=self.train_collection_id,	
            epochs=epochs,
            train_batches=train_batches,
            val_batches=val_batches,
            val_size=val_size,
            lr=lr,
            patience=patience,
            log = self.log,
            log_config=log_config,
            log_epochs=log_epochs,
            run_id=self.model_run_id,
        )

        

    def evaluate_datamodels(self):

        config = DatamodelIndexBasedConfig(
            k = self.k,
            num_models= self.num_models,
            datamodels_path = f"{self.root_path}/datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )

        log_config = None
        if self.log:
            log_config = LogConfig(
                project=self.project_log,
                dir="logs",
                id=f"{self.model_run_id}_evaluate_datamodels_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=f"{self.model_run_id}_evaluate_datamodels",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config),
                    "metrics": "mse"

                },
                tags=self.tags.extend(["evaluation"])
            )

        datamodel = DatamodelsIndexBasedNQPipeline(config)
            
        datamodel.evaluate_test_collections(
                evaluation_id=f"evaluation_{self.model_run_id}_{self.evaluation_metric}",
                collection_name=self.test_collection_id,
                model_id=self.model_run_id,
                log = self.log,
                log_config=log_config,
                metric=self.evaluation_metric
            )


    def get_datamodels_generations(self):


        ## Setup variables
        wiki = pl.read_ipc(self.wiki_path).with_row_index("idx")
        questions = pl.read_ipc(self.questions_path)

        model = GenericInstructModelHF(self.laguage_model_path)
        model_configs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048,
                "max_new_tokens": 10,
                "num_return_sequences": 1
        }

        generations = {}

        ## Load retrieval data
        with open(self.retrieval_path, "r") as f:
            retrieval_data = json.load(f)

        ## Load weights
        weights = torch.load(f"{self.root_path}/datamodels/models/{self.model_run_id}/weights.pt")

        ## Get self.k highest weights
        k_values, k_indices = weights.topk(self.k, dim=1)

        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"datamodels_generations_{self.model_run_id}",
                id = f"datamodels_generations_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,

                },
                tags = self.tags.extend(["datamodels", "generation"]),
            )

            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})


        ## Iterate questions
        for r_idx in range(len(retrieval_data)):

            top_k = [retrieval_data[str(r_idx)][i] for i in k_indices[r_idx]]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                prompt += f"Document[{self.k-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['text'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction=self.instruction, 
                config_params=model_configs
            )

            generations[str(r_idx)] = [str(out["generated_text"]) for out in outputs]

            with open(f"{self.root_path}/generations/{self.datamodels_generation_name}.json", "w") as f:
                json.dump(generations, f)

            if self.log:
                artifact = wandb.Artifact(
                    name="datamodels_generation_data",
                    type="json",
                    description="Datamodels generation daa"
                )

                artifact.add_file(f"{self.root_path}/generations/{self.datamodels_generation_name}.json")
                wandb.log_artifact(artifact)
                wandb.log({
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    "total_duration": (datetime.datetime.now() - start_time).total_seconds(),
                })
                wandb.finish()


    def get_datamodels_retrieval(self):

        """
        This function is used to get the retrieval data from the datamodels.
        
        It takes the model id from the config_datamodels_retrieval, loads the weights
        from the weights.pt file, and then calls the load_weights_to_json function
        to convert the weights into a json file.
        
        The json file is saved in the retrieval folder with the name rag_retrieval_indexes.json.
        The function also saves the retrieval data in the saving path specified in the config.
        """

        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                name=f"datamodels_retrieval_{self.model_run_id}",
                id = f"datmaodels_retieval_{self.model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,

                },
                tags = self.tags.extend(["datamodels", "retrieval"]),
            )

            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})


        model_id = self.model_run_id
        load_weights_to_json(
            f"{self.root_path}/datamodels/models/{model_id}/weights.pt",
            f"{self.root_path}/retrieval/rag_retrieval_indexes.json",
            f"{self.root_path}/retrieval",
            model_id
        )

        if self.log:
            artifact = wandb.Artifact(
                name="datamodels_retrieval_data",
                type="json",
                description="Datamodels retrieval daa"
            )

            artifact.add_file(f"{self.root_path}/retrieval/{model_id}_indexes.json")
            artifact.add_file(f"{self.root_path}/retrieval/{model_id}_weights.json")
            wandb.log_artifact(artifact)
            wandb.log({
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    "total_duration": (datetime.datetime.now() - start_time).total_seconds(),
            })
            wandb.finish()

        




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
            
            case "get_datamodels_generations":
                self.get_datamodels_generations()

            case "create_datamodels_datasets":
                self.create_datamodels_datasets()

            case "run_pre_collections":
                self.run_pre_colections()
            
            case "run_collections":
                self.run_collections()
            
            case "train_datamodels":
                self.train_datamodels()
            
            case "evaluate_datamodels":
                self.evaluate_datamodels()

            case "get_datamodels_retrieval":
                self.get_datamodels_retrieval()
            
            case _:
                raise ValueError(f"Invalid step: {step}")
        
