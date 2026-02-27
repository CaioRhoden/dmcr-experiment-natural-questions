import joblib
from utils.ebm_classifier.DataLoader import DataLoader
from utils.ebm_classifier.EBMTrainer import EBMTrainer
from utils.ebm_classifier import RocAucEvaluator
import numpy as np

class EBMPipeline():

    def __init__(self, data_loader: DataLoader, trainer: EBMTrainer, evaluator: RocAucEvaluator):
        self.data_loader = data_loader
        self.trainer = trainer
        self.evaluator = evaluator

    def run(self, saving_model=None):
        # Load data
        print("Processing EBM data loading...")
        x_train, y_train, x_test, y_test = self.data_loader.load_data()
        
        print("Training EBM models...")
        models = self.trainer.train(x_train, y_train)
        if saving_model is not None:
            joblib.dump(models, saving_model)

        print("Evaluating EBM models...")
        evaluation = self.evaluator.evaluate(models, x_test, y_test)
        
        print("EBM Classifier Pipeline completed.")
        return evaluation