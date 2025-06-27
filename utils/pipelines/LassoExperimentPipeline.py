
from typing import Optional
import polars as pl
from sympy import factor
import torch
import numpy as np
import datetime



from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig

from dmcr.datamodels.models import FactoryLASSOLinearRegressor


from utils.pipelines import RAGBasedExperimentPipeline
from utils.set_random_seed import set_random_seed

class LassoExperimentPipeline(RAGBasedExperimentPipeline):

    def __init__(self, 
                lambda_l1: float,
                tags: list[str] = [],
                seed: Optional[int] = None,
                lm_configs: Optional[dict[str, int|float]] = None,
                log: bool = False,
                root_path: str = ".",
                *args,
                **kwargs,
                 
                 
                ): # Use kwargs to gracefully handle any extra fields):
        super().__init__(*args, **kwargs)
        self.lambda_l1 = lambda_l1
        self.tags = tags
        self.seed = seed
        self.lm_configs = lm_configs if lm_configs is not None else {}
        self.log = log
        self.root_path = root_path

        if seed:
            set_random_seed(seed)
    

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


        factory = FactoryLASSOLinearRegressor(
            in_features=self.size_index,
            out_features=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **{"lambda_l1": self.lambda_l1}  # Unpack lm_configs
        )

        datamodel.train_datamodels(
            model_factory=factory,
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