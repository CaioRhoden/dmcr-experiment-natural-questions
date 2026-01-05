
from curses import start_color
from pyexpat import model
import re
from tracemalloc import start
from typing import Optional
import polars as pl
import os
from regex import B
import torch
import datetime
import faiss
import json
from FlagEmbedding import FlagModel
import wandb

from dmcr.datamodels.setter.IndexBasedSetter import IndexBasedSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
from dmcr.datamodels.pipeline.DatamodelsIndexBasedNQPipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.pipeline.PreCollectionsPipeline import DatamodelsPreCollectionsData, BaseLLMPreCollectionsPipeline, BatchLLMPreCollectionsPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.models import GenericInstructModelHF, GenericInstructBatchHF, GenericVLLMBatch, BaseLLM
from dmcr.evaluators import Rouge_L_evaluator, SquadV2Evaluator, JudgeEvaluator
from dmcr.datamodels.models import LinearRegressor
from dmcr.datamodels.models import FactoryLinearRegressor


from utils.weights_to_json import load_weights_to_json
from utils.set_random_seed import set_random_seed
from utils.context_strategy import nq_context_strategy
from utils.weights_to_json import concat_sorted_tensors

class RAGBasedExperimentPipeline:
    def __init__(
        self,
        # We define a parameter for each field in the Config dataclass
        retrieval_path: str,
        wiki_path: str,
        embedder_path: str,
        vector_db_path: str,
        questions_path: str,
        language_model_path: str,
        project_log: str,
        model_run_id: str,
        k: int,
        size_index: int,
        num_models: int,
        evaluation_metric: str,
        evaluator: str,
        instruction: str, 
        train_samples: int,
        test_samples: int,
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
        datamodels_generation_name: Optional[str] = None,
        batch_size: int = 1,
        attn_implementation: str = "sdpa",
        thinking: bool = False,
        **kwargs, # Use kwargs to gracefully handle any extra fields
    ):
        
        '''
        Initializes the experiment pipeline with individual configuration arguments.
        '''
        # Assign all initialization parameters to instance attributes
        self.seed = seed

        self.retrieval_path = retrieval_path
        self.wiki_path = wiki_path
        self.embedder_path = embedder_path
        self.vector_db_path = vector_db_path
        self.questions_path = questions_path
        self.language_model_path = language_model_path
        self.project_log = project_log
        self.model_run_id: str = model_run_id
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
        self.batch_size = batch_size
        self.attn_implementation = attn_implementation
        self.thinking = thinking



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
        if not os.path.exists(f"{self.root_path}/datamodels/pre_collections/train"):
            os.mkdir(f"{self.root_path}/datamodels/pre_collections/train")
        if not os.path.exists(f"{self.root_path}/datamodels/pre_collections/test"):
            os.mkdir(f"{self.root_path}/datamodels/pre_collections/test")
        if not os.path.exists(f"{self.root_path}/datamodels/collections"):
            os.mkdir(f"{self.root_path}/datamodels/collections")
        if not os.path.exists(f"{self.root_path}/datamodels/evaluations"):
            os.mkdir(f"{self.root_path}/datamodels/evaluations")
        if not os.path.exists(f"{self.root_path}/datamodels/collections/train"):
            os.mkdir(f"{self.root_path}/datamodels/collections/train")
        if not os.path.exists(f"{self.root_path}/datamodels/collections/test"):
            os.mkdir(f"{self.root_path}/datamodels/collections/test")
        if not os.path.exists(f"{self.root_path}/datamodels/models"):
            os.mkdir(f"{self.root_path}/datamodels/models")

            
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
    
    def run_pre_colections(self, 
                           mode: str ="train",
                           start_idx: int = 0,
                           end_idx: int =  -1,
                           checkpoint: int = 50,
                           collection_id: str = "default_collection",
                           model: GenericInstructModelHF | GenericVLLMBatch | GenericInstructBatchHF | None = None
                        ):


        """
        This function creates the pre collections for train and test datasets.
        It uses the DatamodelsIndexBasedNQPipeline to create the pre collections.
        The function takes the instruction, llm, start_idx, end_idx, mode, log,
        log_config, checkpoint, output_column, model_configs, and rag_indexes_path
        as parameters. It returns nothing.
        """
        ### Initiate models
        batch_list = []
        if model is None and self.batch_size == 1:
            model = GenericInstructModelHF(self.language_model_path, attn_implementation=self.attn_implementation, thinking=self.thinking)
        elif model is None and self.batch_size > 1:
            model = GenericVLLMBatch(
                path=self.language_model_path,
                thinking=self.thinking,
                vllm_kwargs={
                    "max_model_len": 32768,
                }
            )
        else:
            raise ValueError("Batch size must be at least 1")



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
                dir=f"{self.root_path}/logs",
                id=f"pre_collection_{collection_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=f"pre_collection_{collection_id}",
                config={
                    "llm": f"{self.language_model_path}",
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,
                    "model_configs": self.lm_configs,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend([mode, "pre_collection"])
            )


        datamodel = DatamodelsIndexBasedNQPipeline(config=config)
        datamodel.set_collections_index()
        datamodel.set_train_dataframes(datamodel.train_set_path)
        datamodel.set_test_dataframes(datamodel.test_set_path)

        print("Start Creating Train Pre Collection")
        pre_collection_data = DatamodelsPreCollectionsData(
            train_collections_idx= datamodel.train_collections_idx,
            test_collections_idx= datamodel.test_collections_idx, 
            train_set= datamodel.train_set,
            test_set= datamodel.test_set,
            datamodels_path= f"{self.root_path}/datamodels"
        )

        ### Normal
        if self.batch_size == 1 and isinstance(model, GenericInstructModelHF):

            pre_collection_pipeline = BaseLLMPreCollectionsPipeline(
                datamodels_data=pre_collection_data,
                mode = mode,
                instruction= self.instruction,
                model = model,
                context_strategy= nq_context_strategy,
                rag_indexes_path=self.retrieval_path,
                output_column = "answers",
                start_idx = start_idx, 
                end_idx = end_idx,  
                checkpoint = checkpoint, 
                log = self.log, 
                log_config = log_config, 
                model_configs = self.lm_configs,
            )

            datamodel.create_pre_collection(pre_collection_pipeline)

        elif self.batch_size > 1:
            print("Using batch pre collection pipeline")
            pre_collection_pipeline = BatchLLMPreCollectionsPipeline(
                datamodels_data=pre_collection_data,
                instruction= self.instruction,
                model = model,
                context_strategy= nq_context_strategy,
                start_idx = start_idx, 
                end_idx = end_idx, 
                mode = mode, 
                log = self.log, 
                log_config = log_config, 
                checkpoint = checkpoint, 
                output_column = "answers",
                model_configs = self.lm_configs,
                rag_indexes_path=self.retrieval_path,
                batch_size=self.batch_size
            )

            datamodel.create_pre_collection(pre_collection_pipeline)

        else:
            raise ValueError("Batch size must be at least 1 and model must be of the correct type")



    def run_collections(self, 
                        mode="train",
                        start_idx: int = 0,
                        end_idx: int = -1,
                        checkpoint: int = 50,
                        collection_id: str = "default_collection"
      ):



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
            judge_model = GenericVLLMBatch(
                path=self.language_model_path,
                thinking=self.thinking,
                vllm_kwargs={
                    "max_model_len": 32768,
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9
                }
            )

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
                model_path=self.language_model_path,
                model_configs = {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_new_tokens": 1024,
                },
                instruction="",
                format_template=format_input,
                thinking=self.thinking,
                judge=judge_model,
                batch_size=self.batch_size


            )

        else:
            raise ValueError(f"Invalid evaluator")


        datamodel = DatamodelsIndexBasedNQPipeline(config)
        datamodel.set_collections_index()
        log_config = None
        if self.log:

            log_config = LogConfig(
                project=self.project_log,
                dir=f"{self.root_path}/logs",
                id=f"{mode}_collections_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=collection_id,
                config={
                    "evaluator": self.evaluator,
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config)
                },
                tags=self.tags.extend([mode, "collections", collection_id])
            )

        print("Start Creating Train Collection")
        datamodel.create_collection(
            evaluator = evaluator,
            mode = mode,
            collection_name = collection_id,
            log = self.log,
            log_config = log_config,
            checkpoint = checkpoint,
            start_idx = start_idx,
            end_idx = end_idx
        )




    def train_datamodels(
            self,
            collection_id: str = "default_collection",
            start_idx: int = 0,
            end_idx: int | None = None,
            checkpoint: int | None = None,
        )-> None:


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
                dir=f"{self.root_path}/logs",
                id=f"test_train_datamoles_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=self.model_run_id,
                config={
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config),

                },
                tags=self.tags.extend(["training", collection_id, self.model_run_id]) # type: ignore
            )



        model_factory = FactoryLinearRegressor(
            in_features=self.size_index,
            out_features=1,
            device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )


        datamodel.train_datamodels(
            model_factory=model_factory,
            collection_name=collection_id,	
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
            checkpoint=checkpoint,
            start_idx=start_idx,
            end_idx=end_idx
        )

        

    def evaluate_datamodels(self, collection_id: str):

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
                dir=f"{self.root_path}/logs",
                id=f"{self.model_run_id}_evaluate_datamodels_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                name=f"{self.model_run_id}_evaluate_datamodels",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "index": "FAISS_L2",
                    "size_index": self.size_index,
                    "datamodel_configs": repr(config),
                    "metrics": "mse"

                },
                tags=self.tags.extend(["evaluation", collection_id, self.model_run_id])
            )

        datamodel = DatamodelsIndexBasedNQPipeline(config)
            
        datamodel.evaluate_test_collections(
                evaluation_id=f"evaluation_{self.model_run_id}_{self.evaluation_metric}",
                collection_name=collection_id,
                model_id=self.model_run_id,
                log = self.log,
                log_config=log_config,
                metric=self.evaluation_metric
            )


    def get_datamodels_generations(
            self,
            datamodels_generation_name,
            model_run_id
        ):


        ## Setup variables
        wiki = pl.read_ipc(self.wiki_path).with_row_index("idx")
        questions = pl.read_ipc(self.questions_path)

        generations = {}

        batch_list = []
        if self.batch_size == 1:
            model = GenericInstructModelHF(self.language_model_path, attn_implementation=self.attn_implementation)
        elif self.batch_size > 1:
            model = GenericVLLMBatch(
                path=self.language_model_path,
                thinking=self.thinking,
                vllm_kwargs={
                    "max_model_len": 32768,
                }
            )

        ## Load retrieval data
        with open(self.retrieval_path, "r") as f:
            retrieval_data = json.load(f)

        ## Load weights
        ### Flag exists "weights.pt"
        weights_dir = f"{self.root_path}/datamodels/models/{model_run_id}"
        exists_weights = os.path.exists(f"{weights_dir}/weights.pt")
        # Flag exists more than on file ending with "_weights.pt"
        exists_more_weights = len([f for f in os.listdir(weights_dir) if f.endswith("_weights.pt")]) >= 1
        if exists_weights:
            weights = torch.load(f"{weights_dir}/weights.pt", weights_only=True)
        elif not exists_weights and exists_more_weights:
            weights = concat_sorted_tensors(weights_dir)
            # assert weights.shape[0] == self.num_models, f"Number of models in the weights tensor ({weights.shape[0]}) does not match the expected number of models ({self.num_models})."

        ## Get self.k highest weights
        k_values, k_indices = weights.topk(self.k, dim=1)

        if self.log:
            start_time = datetime.datetime.now()
            wandb.init(
                project=self.project_log,
                dir=f"{self.root_path}/logs",
                name=f"datamodels_generations_{model_run_id}",
                id = f"datamodels_generations_{model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,

                },
                tags = self.tags.extend(["datamodels", "generation"]),
            )

            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})


        ## Iterate questions
        for r_idx in range(weights.size(0)):

            top_k = [retrieval_data[str(r_idx)][i] for i in k_indices[r_idx]]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                if top_k[doc_idx] >= 0:
                    prompt += f"Document[{self.k-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['text'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            if self.batch_size > 1 and isinstance(model, GenericInstructBatchHF| GenericVLLMBatch):
                if len(batch_list) < self.batch_size:
                    batch_list.append((r_idx, prompt))
                
                if len(batch_list) == self.batch_size or r_idx == (len(retrieval_data) - 1):
                    outputs = model.run(
                        [str(_q[1]) for _q in batch_list],
                        instruction=self.instruction,
                        config_params=self.lm_configs
                    )
                    for i, _q in enumerate(batch_list):
                        generations[f"{_q[0]}"] = []
                        for gen in range(outputs[i].__len__()):
                            generations[f"{_q[0]}"].append(str(outputs[i][gen]["generated_text"]))
                    batch_list = []
                
                    with open(f"{self.root_path}/generations/{datamodels_generation_name}.json", "w") as f:
                        json.dump(generations, f)

            else:
                assert isinstance(model, GenericInstructModelHF)
                ## Generate output
                outputs = model.run(
                    prompt, 
                    instruction=self.instruction, 
                    config_params=self.lm_configs
                )

                generations[str(r_idx)] = [str(out["generated_text"]) for out in outputs]

                with open(f"{self.root_path}/generations/{datamodels_generation_name}.json", "w") as f:
                    json.dump(generations, f)

        if self.log:
            wandb.log({
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "total_duration": (datetime.datetime.now() - start_time).total_seconds(),
            })
            artifact = wandb.Artifact(
                name="datamodels_generation_data",
                type="json",
                description="Datamodels generation daa"
            )

            artifact.add_file(f"{self.root_path}/generations/{self.datamodels_generation_name}.json")
            wandb.log_artifact(artifact)

            wandb.finish()


    def get_datamodels_retrieval(self, model_run_id: str):

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
                dir=f"{self.root_path}/logs",
                name=f"datamodels_retrieval_{model_run_id}",
                id = f"datmaodels_retieval_{model_run_id}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
                config={
                    "gpu": f"{torch.cuda.get_device_name(0)}",
                    "size_index": self.size_index,

                },
                tags = self.tags.extend(["datamodels", "retrieval"]),
            )

            wandb.log({"start_time": start_time.strftime('%Y-%m-%d_%H-%M-%S')})


        load_weights_to_json(
            f"{self.root_path}/datamodels/models/{model_run_id}",
            self.retrieval_path,
            f"{self.root_path}/retrieval",
            model_run_id,
            num_models=self.num_models
        )

        if self.log:
            wandb.log({
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    "total_duration": (datetime.datetime.now() - start_time).total_seconds(),
            })
            artifact = wandb.Artifact(
                name="datamodels_retrieval_data",
                type="json",
                description="Datamodels retrieval daa"
            )

            artifact.add_file(f"{self.root_path}/retrieval/{model_run_id}_indexes.json")
            artifact.add_file(f"{self.root_path}/retrieval/{model_run_id}_weights.json")
            wandb.log_artifact(artifact)

            wandb.finish()

    