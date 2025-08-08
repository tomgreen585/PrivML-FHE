import sys
from fc_data_loader import ML_Data_Loader
from fc_preprocessing import ML_Preprocessing
from fc_model import ML_Model
from fc_training import ML_Training_Class
from fc_testing import ML_Testing_Class
from fc_save import ML_Saving_Model
from fc_evaluation import Ml_Metrics_Evaluation
from fc_config import (
    DATASET_PATH, IMAGE_SIZE, TRAIN_RATIO, VAL_RATIO,
    SEED, SAMPLE_OUTPUT_COUNT, EPOCHS, BATCH_SIZE, LEARNING_RATE
)

# main.py
# - Simple orchestrator script that ties everything together
# - Loads config
# - Trains the model
# - Evaluates it
# - Runs encryption tests

class ML_Main:
    def __init__(self):
        self.images = []
        self.dataset_path = DATASET_PATH
        self.image_size = IMAGE_SIZE
        self.train_ratio = TRAIN_RATIO
        self.val_ratio = VAL_RATIO
        self.seed = SEED
        self.sample_output_count = SAMPLE_OUTPUT_COUNT
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        
        self.loading_dataset = False
        self.display_dataset_metrics = False
        self.completed_loading_data = False
        self.completed_loading_dataset = False
        self.completed_preprocessing_steps = False
        self.completed_training_model = False
        self.completed_testing_model = False
        self.completed_saving_model = False
        self.completed_ml_pipeline = False
        
    def data_loader(self, dataset_path, image_size):
        print("[INFO] Starting to Load Data")
        
        loader = ML_Data_Loader(dataset_path, image_size)
        
        if self.loading_dataset:
            loader.loading_dataset()
            x_data, y_data = loader.get_dataset_with_bounding_boxes()
            if self.display_dataset_metrics:
                loader.display_dataset_metrics()
            self.completed_loading_dataset = True
        else:
            print("[ERR] Failed to load data")
            exit(1)
        
        self.completed_loading_data = True
        return x_data, y_data
            
    def preprocess_data(self, x_data, y_data, train_ratio, val_ratio, seed, run_type):
        print("[INFO] Performing Preprocessing Steps on the Data")
        
        preprocess = ML_Preprocessing()
       
        x_train, y_train, x_val, y_val, x_test, y_test = preprocess.preprocessing_steps(x_data, y_data, train_ratio, val_ratio, seed, run_type)
        
        self.completed_preprocessing_steps = True
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def ml_training_loop(self, model, x_train, y_train, x_val, y_val):
        print("[INFO] Starting ML Training Loop")
        
        training = ML_Training_Class()
        
        model = training.train_model(model, x_train, y_train, x_val, y_val, self.epochs, self.batch_size, self.learning_rate)
        
        self.completed_training_model = True
        return model
        
    def ml_testing_loop(self, model, x_test, y_test, run_type):
        print("[INFO] Starting ML Testing Loop")
        
        testing = ML_Testing_Class()
        
        testing.test_model(model, x_test, y_test, self.sample_output_count, run_type)
        
        self.completed_testing_model = True
        
    def ml_saving_loop(self, model):
        print("[INFO] Starting ML Saving Loop")
        
        save = ML_Saving_Model()
        save.save_ml_model(model)
        
        self.completed_saving_model = True
    
    def ml_main_pipeline(self, run_type):
        print("[INFO] Running ML Pipeline")
        
        if run_type == "Testing":
            self.loading_dataset = True
            self.display_dataset_metrics = True
            x_data, y_data = self.data_loader(self.dataset_path, self.image_size)
            x_train, y_train, x_val, y_val, x_test, y_test = self.preprocess_data(x_data, y_data, self.train_ratio, self.val_ratio, self.seed, run_type)
            model = ML_Model()
            model = self.ml_training_loop(model, x_train, y_train, x_val, y_val)
            self.ml_testing_loop(model, x_test, y_test, run_type)
            
        elif run_type == "Final_Model":
            self.loading_dataset = True
            self.display_dataset_metrics = False
            x_data, y_data = self.data_loader(self.dataset_path, self.image_size)
            x_train, y_train, x_val, y_val, x_test, y_test = self.preprocess_data(x_data, y_data, self.train_ratio, self.val_ratio, self.seed, run_type)
            model = ML_Model()
            model = self.ml_training_loop(model, x_train, y_train, x_val, y_val)
            self.ml_testing_loop(model, x_test, y_test, run_type)
            self.ml_saving_loop(model)
        
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