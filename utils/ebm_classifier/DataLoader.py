from re import S
import polars as pl
import numpy as np



class DataLoader():

    def __init__(self, load_path:str, subfolders= ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]):
        self.load_path = load_path
        self.subfolders = subfolders

    def load_data(self, num_models=500):
        '''
        This method loads the train and test data for the EBM pipeline. It reads the data from the specified path, processes it, 
        and returns the training and testing datasets for both groundtruth and judge supervised experiments.

        This method assumes that inside the load_path, there are subfolders for each experiment, and inside each subfolder, 
        there are "train.feather" and "test.feather" files containing the data. The method reads these files, concatenates them into a single DataFrame, 
        and then splits the data into training and testing sets for each experiment. Finally, it reshapes the data to ensure that it is in the correct format 
        for training the EBM model.

        Args:
            num_models (int): The number of models in the dataset. This is used to reshape the data correctly. Default is 500.
        
        Returns:
            X_train (list): A list of training input data for each experiment, the shape is (num_exp_subfolders, num_models, num_train_samples_per_model).
            y_train (list): A list of training labels for each experiment, the shape is (num_exp_subfolders, num_models, num_train_samples_per_model).
            X_test (list): A list of testing input data for each experiment, the shape is (num_exp_subfolders, num_models, num_test_samples_per_model).
            y_test (list): A list of testing labels for each experiment, the shape is (num_exp_subfolders, num_models, num_test_samples_per_model).
        '''
        
        df = []

        for exp in self.subfolders:
            for split in ["train", "test"]:
                load_df = pl.read_ipc(f"{self.load_path}/{exp}/{split}.feather")
                load_df = load_df.with_columns([
                    pl.lit(exp).alias("experiment"),
                    pl.lit(split).alias("split"),
                ])
                df.append(load_df)
        
        df = pl.concat(df)


        train = df.filter((pl.col("split") == "train"))
        test = df.filter((pl.col("split") == "test"))
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []



        for exp in self.subfolders:
            train_exp = train.filter(pl.col("experiment") == exp)
            test_exp = test.filter(pl.col("experiment") == exp)
            
            X_train_exp = train_exp.select("input").to_numpy()
            X_train_exp = np.array([i[0] for i in X_train_exp])
            y_train_exp = train_exp.select("evaluation").to_numpy()

            X_test_exp = test_exp.select("input").to_numpy()
            X_test_exp = np.array([i[0] for i in X_test_exp])
            y_test_exp = test_exp.select("evaluation").to_numpy()

            # Reshape groundtruth data: (500*2000,) -> (500, 2000)
            X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped = [], [], [], []
            for i in range(num_models):
                # Take every 500th element starting from index i
                X_train_reshaped.append(X_train_exp[i::num_models])
                y_train_reshaped.append(y_train_exp[i::num_models])
                X_test_reshaped.append(X_test_exp[i::num_models])
                y_test_reshaped.append(y_test_exp[i::num_models])
            

            X_train.append(X_train_reshaped)
            y_train.append(y_train_reshaped)
            X_test.append(X_test_reshaped)
            y_test.append(y_test_reshaped)

        return X_train, y_train, X_test, y_test


