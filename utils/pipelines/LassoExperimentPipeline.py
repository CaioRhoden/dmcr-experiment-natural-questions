
import polars as pl
import torch
import numpy as np
import datetime



from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig

from dmcr.datamodels.models import LASSOLinearRegressor


from utils.pipelines import RAGBasedExperimentPipeline

class LassoExperimentPipeline(RAGBasedExperimentPipeline):

    def __init__(self,
        # We define a parameter for each field in the Config dataclass
        seed: int,
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
        lambda_l1: float,
        tags: list[str] = [],
        model_id_retrieval: str = "",
        **kwargs, # Use kwargs to gracefully handle any extra fields
    ):
        
        super().__init__(
            seed=seed,
            retrieval_path=retrieval_path,
            wiki_path=wiki_path,
            embeder_path=embeder_path,
            vector_db_path=vector_db_path,
            questions_path=questions_path,
            laguage_model_path=laguage_model_path,
            project_log=project_log,
            model_run_id=model_run_id,
            train_collection_id=train_collection_id,
            test_collection_id=test_collection_id,
            k=k,
            size_index=size_index,
            num_models=num_models,
            evaluation_metric=evaluation_metric,
            evaluator=evaluator,
            instruction=instruction,
            train_samples=train_samples,
            test_samples=test_samples,
            train_start_idx=train_start_idx,
            train_end_idx=train_end_idx,
            test_start_idx=test_start_idx,
            test_end_idx=test_end_idx,
            train_checkpoint=train_checkpoint,
            test_checkpoint=test_checkpoint,    
            epochs=epochs, 
            lr=lr,
            train_batches=train_batches,
            val_batches=val_batches,
            val_size=val_size,
            patience=patience,
            log_epochs=log_epochs,
            tags=tags,
            model_id_retrieval=model_id_retrieval,
        )

        self.lambda_l1 = lambda_l1
    

    def train_datamodels(self):

        """
        Trains the datamodels using the specified configuration and parameters.

        This function initializes the datamodel configuration and pipeline, 
        sets up logging, and trains the datamodels using a LASSOLinearRegressor model.
        It uses the configurations provided during initialization to specify
        training parameters such as epochs, learning rate, batch sizes, and more.

        Attributes:
            epochs (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
            train_batches (int): Number of training batches.
            val_batches (int): Number of validation batches.
            val_size (float): Size of the validation set.
            patience (int): Patience for early stopping.
            log_epochs (int): Interval of epochs to log training progress.

        Creates:
            DatamodelsIndexBasedNQPipeline: Pipeline for managing datamodels.
            LogConfig: Configuration for logging the training process.
            LASSOLinearRegressor: Model used for training with L1 regularization.

        Executes:
            Trains the datamodels with the specified configurations and logs the progress.
        """

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
            datamodels_path = "datamodels",
            train_set_path=self.wiki_path,
            test_set_path=self.questions_path
        )



        datamodel = DatamodelsIndexBasedNQPipeline(config)

        log_config = LogConfig(
            project=self.project_log,
            dir="logs",
            id=f"test_train_datamoles_{str(datetime.datetime.now)}",
            name=self.model_run_id,
            config={
                "gpu": f"{torch.cuda.get_device_name(0)}",
                "index": "FAISS_L2",
                "size_index": self.size_index,
                "datamodel_configs": repr(config),

            },
            tags=self.tags.extend(["training"])
        )


        model = LASSOLinearRegressor(
            in_features=self.size_index,
            out_features=1,
            lambda_l1=self.lambda_l1,
            device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )

        datamodel.train_datamodels(
            model=model,
            collection_name=self.train_collection_id,	
            epochs=epochs,
            train_batches=train_batches,
            val_batches=val_batches,
            val_size=val_size,
            lr=lr,
            patience=patience,
            log=True,
            log_config=log_config,
            log_epochs=log_epochs,
            run_id=self.model_run_id,
        )