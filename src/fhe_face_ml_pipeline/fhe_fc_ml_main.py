import sys
from fhe_fc_data_loader import ML_Data_Loader
from fhe_fc_preprocessing import ML_Preprocessing
from fhe_fc_plain_model import ML_Model
from fhe_fc_fhe_model import FHE_Model
from fhe_fc_plain_training import ML_Training_Class
from fhe_fc_plain_testing import ML_Testing_Class
from fhe_fc_enc_testing import ML_FHE_Testing_Class
from fhe_fc_save import ML_Saving_Model
from fhe_fc_evaluation import Ml_Metrics_Evaluation

"""
main.py

Runs the main pipeline. Performs data loading, preprocessing, 
training, testing, evaluation and saving.

Runs in two different modes: -t ("Testing" - train, test, but don't save model + visualise plots) 
and -f ("Final_Model" - train, test, saves model + doesnt visualize plots).
"""

class ML_Main:
    """
    Main pipeline handler for training the facial recognition ML model.
    """
    def __init__(self):
        self.loading_dataset = False
        self.display_dataset_metrics = False
        self.completed_loading_data = False
        self.completed_loading_dataset = False
        self.completed_preprocessing_steps = False
        self.completed_training_model = False
        self.completed_testing_model = False
        self.completed_saving_model = False
        self.completed_ml_pipeline = False
        
    def data_loader(self):
        """
        Loads and annotates dataset using face recognition library and 
        returns input/output for model training.

        Returns:
        x_data (np array): Original images
        y_data (np array): Target labeled images
        """
        print("[INFO] Starting to Load Data")
        
        loader = ML_Data_Loader()
        
        if self.loading_dataset:
            loader.load_dataset()
            x_data, y_data = loader.generating_model_datasets()
            if self.display_dataset_metrics:
                loader.display_dataset_metrics()
            self.completed_loading_dataset = True
        else:
            print("[ERR] Failed to load data")
            exit(1)
        
        self.completed_loading_data = True
        return x_data, y_data
            
    def preprocess_data(self, x_data, y_data, run_type):
        """
        Applies preprocessing steps (e.g., normalization, train/val/test split).

        Args:
        x_data (np array): Input images
        y_data (np array): Target images
        run_type (str): "Testing" or "Final_Model"

        Returns:
        train/val/test/encryption splits for x and y
        """
        print("[INFO] Performing Preprocessing Steps on the Data")
        
        preprocess = ML_Preprocessing()
       
        x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test, x_enc_test, y_enc_test = preprocess.preprocessing_steps(x_data, y_data, run_type)
        
        self.completed_preprocessing_steps = True
        return x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test, x_enc_test, y_enc_test
    
    def ml_plaintext_training_loop(self, model, x_train, y_train, x_val, y_val):
        """
        Trains the ML model using the training and validation sets.

        Args:
        model (ML_Model): The initialized model
        x_train, y_train, x_val, y_val: Training/validation datasets

        Returns:
        model (ML_Model): trained model instance
        """
        print("[INFO] Starting ML Training Loop")
        
        training = ML_Training_Class()
        
        model = training.train_model(model, x_train, y_train, x_val, y_val)
        
        self.completed_training_model = True
        return model
        
    def ml_plaintext_testing_loop(self, model, x_test, y_test, run_type):
        """
        Tests the trained model on the test set.

        Args:
        model (ML_Model): trained model
        x_test, y_test (np.ndarray): Test data and labels
        run_type (str): Execution context ("Testing" or "Final_Model")
        """
        print("[INFO] Starting ML Testing Loop")
        
        testing = ML_Testing_Class()
        
        testing.test_model(model, x_test, y_test, run_type)
        
        self.completed_testing_model = True
        
    def ml_encrypted_testing_loop(self, plain_model, enc_model, x_test, y_test, run_type):
        """
        Tests the trained model wrapped in the initialized FHE_Model on the encryption set.

        Args:
        plain_model (ML_Model): trained model
        enc_model (FHE_Model): 
        x_test, y_test (np.ndarray): Test data and labels
        run_type (str): Execution context ("Testing" or "Final_Model")
        """
        print("[INFO] Starting ML Encrypted Testing Loop")
        
        enc_testing = ML_FHE_Testing_Class()
        
        enc_testing.testing_pipeline(plain_model, enc_model, x_test, y_test, run_type)
        
        self.completed_encrypted_testing_model = True
        
    def ml_saving_loop(self, model):
        """
        Saves the trained model to directory.
        """
        print("[INFO] Starting ML Saving Loop")
        
        save = ML_Saving_Model()
        save.save_ml_model(model)
        
        self.completed_saving_model = True
    
    def ml_main_pipeline(self, run_type):
        """
        Method that runs the ML pipeline end-to-end.

        Args:
        run_type (str): "Testing" or "Final_Model"
        """
        print("[INFO] Running ML Pipeline")
        
        if run_type == "Testing":
            self.loading_dataset = True
            self.display_dataset_metrics = True
            
            x_data, y_data = self.data_loader()
            x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test, x_enc_test, y_enc_test = self.preprocess_data(x_data, y_data, run_type)
            
            plain_model = ML_Model()
            plain_model = self.ml_plaintext_training_loop(plain_model, x_plain_train, y_plain_train, x_plain_val, y_plain_val)
            self.ml_plaintext_testing_loop(plain_model, x_plain_test, y_plain_test, run_type)
            
            fhe_model = FHE_Model(plain_model.eval())
            self.ml_encrypted_testing_loop(plain_model, fhe_model, x_enc_test, y_enc_test, run_type)
            
        elif run_type == "Final_Model":
            self.loading_dataset = True
            self.display_dataset_metrics = False
            
            x_data, y_data = self.data_loader()
            x_plain_train, y_plain_train, x_plain_val, y_plain_val, x_plain_test, y_plain_test, x_enc_test, y_enc_test = self.preprocess_data(x_data, y_data, run_type)
            
            plain_model = ML_Model()
            plain_model = self.ml_plaintext_training_loop(plain_model, x_plain_train, y_plain_train, x_plain_val, y_plain_val)
            self.ml_plaintext_testing_loop(plain_model, x_plain_test, y_plain_test, run_type)
            
            fhe_model = FHE_Model(plain_model.eval())
            self.ml_encrypted_testing_loop(plain_model, fhe_model, x_enc_test, y_enc_test, run_type)
            self.ml_saving_loop(plain_model)
        
        model_id = Ml_Metrics_Evaluation.save_ml_metrics_csv()
        Ml_Metrics_Evaluation.create_ml_report(model_id)
        self.completed_ml_pipeline = True   
        print("[INFO] Finished Running ML Pipeline")
                            
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f'Command arguments: {sys.argv[1]}')
        if sys.argv[1] == "-t":
            run_type = "Testing"
            ml = ML_Main()
            ml.ml_main_pipeline(run_type)
        elif sys.argv[1] == "-f":
            run_type = "Final_Model"
            ml = ML_Main()
            ml.ml_main_pipeline(run_type)
        else:
            print("[ERR] Unknown argument. Use -t for Testing or -f for Final_Model.")
            sys.exit(1)
    else:
        print("[ERR] No run type argument provided. Use -t for Testing or -f for Final_Model.")
        sys.exit(1)