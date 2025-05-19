
import polars as pl
import os
import torch
import numpy as np
import random
from dmcr.datamodels.setter.IndexBasedSetter import IndexBasedSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
from dmcr.models import GenericInstructModelHF
from dmcr.evaluators import Rouge_L_evaluator
import datetime
import faiss
import json
from FlagEmbedding import FlagModel

import yaml


class RAGBasedExperimentPipeline:

    def __init__(self, config_path: str):
        config = yaml.safe_load(open(config_path, "r"))
        self.config = config["global_config"]
        self.config_pre_collections = config["pre_collections_config"]
        self.config_datamodels_training = config["datamodels_training_config"]

    def set_random_seed(self):

        np.random.seed(self.config["seed"])
        random.seed(self.config["seed"])
        pl.set_random_seed(self.config["seed"])

        # PyTorch
        torch.manual_seed(self.config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config["seed"])
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")



    def setup(self):

        """
        Setup for the experiment.

        This function downloads the 100 question golden dataset, writes it to questions.feather and creates the retrieval, generations, results and datamodels folders.
        """
        ## Create structure
        os.mkdir("retrieval")
        os.mkdir("generations")
        os.mkdir("results")
        os.mkdir("datamodels")

        ## Create Datamodels Structure
        os.mkdir("datamodels/datasets")
        os.mkdir("datamodels/pre_collections")
        os.mkdir("datamodels/collections")
        os.mkdir("datamodels/models")


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

        df = pl.read_ipc(self.config["questions_path"])

        ### Load faiss indices
        index = faiss.read_index(self.config["vector_db_path"])
        # ip_index = faiss.read_index(IP_FAISS_INDEX_PATH)
        embedder = FlagModel(self.config["embeder_path"], devices=["cuda:0"], use_fp16=True)



        ### Iterate questions
        for idx in range(len(df)):

            question = df[idx]["question"].to_numpy().flatten()[0]
            query_embedding = embedder.encode(
                [question],
                convert_to_numpy=True,
            )
            query_embedding = query_embedding.astype('float32').reshape(1, -1)

            ### Get l2 and ip neighbors
            scores, ids = index.search(query_embedding, 100)
            # ip_ids, ip_scores = ip_index.search(query_embedding, 100)

            retrieval_indexes[idx] = ids.tolist()[0]
            retrieval_distances[idx] = scores.tolist()[0]
            # retrieval_data["ip"][idx] = (ip_ids.tolist()[0], ip_scores.tolist()[0])

        ## Save into json
        with open("retrieval/rag_retrieval_indexes.json", "w") as f:
            json.dump(retrieval_indexes, f)

        with open("retrieval/rag_retrieval_distances.json", "w") as f:
            json.dump(retrieval_distances, f)

    def get_rag_generations(self):


        ## Setup variables
        wiki = pl.read_ipc(self.config["wiki_path"]).with_row_index("idx")
        questions = pl.read_ipc(self.config["questions_path"])

        model = GenericInstructModelHF(self.config["laguage_model_path"])
        model_configs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048,
                "max_new_tokens": 10,
                "num_return_sequences": 5
        }

        generations = {}

        ## Load retrieval data
        with open(self.config["retrieval_path"], "r") as f:
            retrieval_data = json.load(f)

        print(retrieval_data)
        ## Iterate questions
        for r_idx in range(len(retrieval_data)):

            top_k = retrieval_data[f"{r_idx}"][0:self.config['k']]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                prompt += f"Document[{self.config['k']-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents ", 
                config_params=model_configs
            )

            generations[f"{r_idx}"] = [str(out["generated_text"]) for out in outputs]

            with open("generations/rag_generations.json", "w") as f:
                json.dump(generations, f)

        ## Save into json
        
            
    def create_datamodels_datasets(self):
        """
        This function creates two .h5 files, training and testing, with respective sizes train_samples and test_samples
        Each element of the dataset corresponds in array of k samples going from [0, size_index)
        These elements represents the position on the RAG dict, as the index for each sample may vary the position in the relative top-size_indez retrieved samples will be
        the same
        """

        DATASET_PATH = "datamodels"
        setter_config = IndexBasedSetterConfig(
            save_path=DATASET_PATH,
            size_index=self.config["size_index"],
            k=self.config['k'],
            train_samples= self.config_pre_collections["train_samples"],
            test_samples= self.config_pre_collections["test_samples"]
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
        model = GenericInstructModelHF(self.config["laguage_model_path"])

        model_configs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048,
                "max_new_tokens": 10
        }

        config = DatamodelIndexBasedConfig(
            k = self.config['k'],
            num_models= self.config["num_models"],
            datamodels_path = "datamodels",
            train_set_path=self.config["wiki_path"],
            test_set_path=self.config["questions_path"]
        )

        train_log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"pre_collection_{self.config['train_collection_id']}_{str(datetime.datetime.now)}",
            name=f"pre_collection_{self.config['train_collection_id']}",
            config={
                "llm": f"{self.config['laguage_model_path']}",
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "size_index": self.config["size_index"],
                "model_configs": model_configs,
                "datamodel_configs": repr(config)
            },
            tags=self.config_pre_collections["tags"].extend(["train", "pre_collection"])
        )

        test_log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"pre_collection_{self.config['test_collection_id']}_{str(datetime.datetime.now)}",
            name=f"pre_collection_{self.config['test_collection_id']}",
            config={
                "llm": f"{self.config['laguage_model_path']}",
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "size_index": self.config["size_index"],
                "model_configs": model_configs,
                "datamodel_configs": repr(config)
            },
            tags=self.config_pre_collections["tags"].extend(["test", "pre_collection"])
            
        )



        datamodel = DatamodelsIndexBasedNQPipeline(config=config)

        print("Start Creating Train Pre Collection")
        datamodel.create_pre_collection(
            instruction= self.config_pre_collections["instruction"],
            llm = model,
            start_idx = self.config_pre_collections["train_start_idx"], 
            end_idx = self.config_pre_collections["train_end_idx"], 
            mode = "train", 
            log = True, 
            log_config = train_log_config, 
            checkpoint = self.config_pre_collections["train_checkpoint"], 
            output_column = "answers",
            model_configs = model_configs,
            rag_indexes_path="retrieval/rag_retrieval_indexes.json"
        )

        print("Start Creating Test Pre Collection")
        datamodel.create_pre_collection(
            instruction= self.config_pre_collections["instruction"],
            llm = model,
            start_idx = self.config_pre_collections["test_start_idx"], 
            end_idx = self.config_pre_collections["test_end_idx"], 
            mode = "test", 
            log = True, 
            log_config = test_log_config, 
            checkpoint = self.config_pre_collections["test_checkpoint"], 
            output_column = "answers",
            model_configs = model_configs,
            rag_indexes_path="retrieval/rag_retrieval_indexes.json"
        )


    def run_collections(self):



        config = DatamodelIndexBasedConfig(
            k = self.config['k'],
            num_models= self.config["num_models"],
            datamodels_path = "datamodels",
            train_set_path=self.config["wiki_path"],
            test_set_path=self.config["questions_path"]
        )


        evaluator = Rouge_L_evaluator()

        datamodel = DatamodelsIndexBasedNQPipeline(config)

        test_log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"test_collections_{str(datetime.datetime.now)}",
            name=self.config["test_collection_id"],
            config={
                "evaluator": "Rouge-L",
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "index": "FAISS_L2",
                "size_index": self.config["size_index"],
                "datamodel_configs": repr(config)
            },
            tags=["test", "collections", "FAISS_L2", "top_100"]
        )

        train_log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"train_collections_{str(datetime.datetime.now)}",
            name=self.config["train_collection_id"],
            config={
                "evaluator": "Rouge-L",
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "index": "FAISS_L2",
                "size_index": self.config["size_index"],
                "datamodel_configs": repr(config)
            },
            tags=["test", "collections", "FAISS_L2", "top_100"]
        )

        print("Start Creating Train Collection")
        datamodel.create_collection(
            evaluator = evaluator,
            mode = "train",
            collection_name = self.config["train_collection_id"],
            log = True,
            log_config = train_log_config
        )


        print("Start Creating Test Collection")
        datamodel.create_collection(
            evaluator = evaluator,
            mode = "test",
            collection_name = self.config["test_collection_id"],	
            log = True,
            log_config = test_log_config
        )


    def train_datamodels(self):

        epochs = self.config_datamodels_training["epochs"]
        lr = self.config_datamodels_training["lr"]
        train_batches = self.config_datamodels_training["train_batches"]
        val_batches = self.config_datamodels_training["val_batches"]
        val_size = self.config_datamodels_training["val_size"]
        patience = self.config_datamodels_training["patience"]
        log_epochs = self.config_datamodels_training["log_epochs"]


        config = DatamodelIndexBasedConfig(
            k = self.config['k'],
            num_models= self.config["num_models"],
            datamodels_path = "datamodels",
            train_set_path=self.config["wiki_path"],
            test_set_path=self.config["questions_path"]
        )



        datamodel = DatamodelsIndexBasedNQPipeline(config)

        log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"test_train_datamoles_{str(datetime.datetime.now)}",
            name=self.config['model_run_id'],
            config={
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "index": "FAISS_L2",
                "size_index": self.config["size_index"],
                "datamodel_configs": repr(config),
                "training_configs": self.config_datamodels_training

            },
            tags=self.config_pre_collections["tags"].extend(["training"])
        )

        datamodel.train_datamodels(
            collection_name=self.config["train_collection_id"],	
            epochs=epochs,
            train_batches=train_batches,
            val_batches=val_batches,
            val_size=val_size,
            lr=lr,
            patience=patience,
            log=True,
            log_config=log_config,
            log_epochs=log_epochs,
            run_id=self.config['model_run_id'],
        )

        

    def evaluate_datamodels(self):

        config = DatamodelIndexBasedConfig(
            k = self.config['k'],
            num_models= self.config["num_models"],
            datamodels_path = "datamodels",
            train_set_path=self.config["wiki_path"],
            test_set_path=self.config["questions_path"]
        )

        log_config = LogConfig(
            project="subpartition-datamodels-rag",
            dir="logs",
            id=f"{self.config['model_run_id']}_evaluate_datamodels_{str(datetime.datetime.now)}",
            name=f"{self.config['model_run_id']}_evaluate_datamodels",
            config={
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "index": "FAISS_L2",
                "size_index": self.config["size_index"],
                "datamodel_configs": repr(config),
                "metrics": "mse"

            },
            tags=["test", "evaluation", "FAISS_L2", "top_100"]
        )

        datamodel = DatamodelsIndexBasedNQPipeline(config)
            
        datamodel.evaluate_test_collections(
                evaluation_id=f"evaluation_{self.config['model_run_id']}_{self.config['evaluation_metric']}",
                collection_name=self.config["test_collection_id"],
                model_id=self.config['model_run_id'],
                log=True,
                log_config=log_config,
                metric=self.config["evaluation_metric"]
            )


    def get_datamodels_generations(self):


        ## Setup variables
        wiki = pl.read_ipc(self.config["wiki_path"]).with_row_index("idx")
        questions = pl.read_ipc(self.config["questions_path"])

        model = GenericInstructModelHF(self.config["laguage_model_path"])
        model_configs = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 2048,
                "max_new_tokens": 10,
                "num_return_sequences": 5
        }

        generations = {}

        ## Load retrieval data
        with open(self.config["retrieval_path"], "r") as f:
            retrieval_data = json.load(f)

        ## Load weights
        weights = torch.load(f"datamodels/models/{self.config['model_run_id']}/weights.pt")

        ## Get self.config["k highest weights
        k_values, k_indices = weights.topk(self.config['k'], dim=1)


        ## Iterate questions
        for r_idx in range(len(retrieval_data)):

            top_k = [retrieval_data[f"{r_idx}"][i] for i in k_indices[r_idx]]
            docs = wiki.filter(pl.col("idx").is_in(top_k))


            ## Generate prompt
            prompt = "Documents: \n"
            for doc_idx in range(len(top_k)-1, -1, -1):
                prompt += f"Document[{self.config['k']-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}\n\n"
            prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

            ## Generate output
            outputs = model.run(
                prompt, 
                instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents ", 
                config_params=model_configs
            )

            generations[f"{r_idx}"] = [str(out["generated_text"]) for out in outputs]

            with open("generations/datamodels_generations.json", "w") as f:
                json.dump(generations, f)


    def invoke_pipeline_stpe(self, step: str):

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
        
