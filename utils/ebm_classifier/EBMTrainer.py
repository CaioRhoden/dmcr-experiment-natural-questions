from interpret.glassbox import ExplainableBoostingClassifier
import polars as pl
from tqdm import tqdm



class EBMTrainer:

    def __init__(self, model_configs: dict):
        self.model_configs = model_configs

    def train(self, X_train, y_train) -> list[ExplainableBoostingClassifier]:

        """
        Train EBM models for each X_train and y_train pair of each subfolder.
        Args:
            X_train (list): A list of training input data for each experiment, the shape is (num_exp_subfolders, num_models, num_train_samples_per_model).
            y_train (list): A list of training labels for each experiment, the shape is (num_exp_subfolders, num
        
        Returns:
            models (list): A list of trained EBM models for each experiment and model, the shape is (num_exp_subfolders, num_models).
        """


        num_subfolders = len(X_train)
        num_models = len(X_train[0])

        models = []
        with tqdm(total=num_subfolders * num_models, desc="Training models") as pbar:
            for i in range(num_subfolders):
                _subfolder_models = []
                for j in range(num_models):
                    _X = X_train[i][j]
                    _y = y_train[i][j]
                    model = ExplainableBoostingClassifier(**self.model_configs)
                    model.fit(_X, _y)
                    _subfolder_models.append(model)

                    pbar.update(1)
                models.append(_subfolder_models)


                
                    
        return models