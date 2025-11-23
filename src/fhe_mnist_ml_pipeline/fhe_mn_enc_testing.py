import torch
import time
import tenseal as ts
from fhe_mn_evaluation import Ml_Metrics_Evaluation
from fhe_mn_config import (
    SCHEME_TYPE, ENCRYPTED_LOSS_FUNCTION, 
    BITS_SCALE, POLY_MODULUS_DEGREE, 
    ST_KEY_GENERATION, END_DECRYPTION_STABILITY, COEFF_MOD_BIT_SIZES
)

"""
enc_testing.py

For running predictions on new encrypted inputs after training. Useful for demonstraing practical use 
of the encrypted model in deployment. Sends performance metrics to evaluation.py to visualize/track.
"""

class ML_FHE_Testing_Class:
    
    """
    Handles testing and evaluation of encrypted models using CKKS homomorphic encryption.
    Performs encrypted inference on a sample batch and logs evaluation metrics.
    
    scheme_type: homomorphic encryption scheme (CKKS)
    loss_function: loss function for evaluating encrypted output
    bits_scale: scaling factor for CKKS
    poly_modulus_degree: polynomial modulus degree
    start_key_generation: start coefficient size for encryption context
    end_decryption_stability: final coefficient size
    """
    def __init__(self):
        self.scheme_type = SCHEME_TYPE
        self.loss_function = ENCRYPTED_LOSS_FUNCTION
        self.bits_scale = BITS_SCALE
        self.poly_modulus_degree = POLY_MODULUS_DEGREE
        self.start_key_generation = ST_KEY_GENERATION
        self.end_decryption_stability = END_DECRYPTION_STABILITY
        self.coeff_bit_sizes = COEFF_MOD_BIT_SIZES
        self.testing_completed = False
        
    def test_enc_model(self, model_type, encryption_context, enc_model, x_test_enc, y_test, loss_function, kernel_shape, stride):
        """
        Runs encrypted inference on a number of test samples and evaluates performance.
        
        Args:
        enc_model: Initialised FHE model
        x_test_enc: plaintext input
        y_test: plaintext ground truth
        kernel_shape: plaintext model kernel shape value
        stride: plaintext model stride value
        encryption_context: initialized CKKS encryption context
        run_type (str): "Training" or "Final_Model"
        model_type (str): "FHE_Model"
        """
        
        print("[INFO] Starting Encrypted Inference Test")

        test_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        losses = []

        total_start_time = time.time()
        print(f"[INFO] Total samples to process {len(x_test_enc)}")
        
        for idx in range(len(x_test_enc)):
            
            if idx % 50 == 0:
                print(f"[INFO] Processing sample {idx}/{len(x_test_enc)}")
                #log CPU/memory usage
                Ml_Metrics_Evaluation.log_resource_usage(model_type)

            sample = x_test_enc[idx].squeeze().tolist()
            label = int(y_test[idx])

            #encode sample using im2col and CKKS
            x_enc, windows_nb = ts.im2col_encoding(
                encryption_context, sample, kernel_shape[0],
                kernel_shape[1], stride
            )
            
            #run inference on encrypted model
            enc_output = enc_model(x_enc, windows_nb)

            #decrypt prediction
            output = enc_output.decrypt()
            output = torch.tensor(output).view(1, -1)
            
            #compute loss
            loss = loss_function(output, torch.tensor([label]))
            test_loss += loss.item()
            losses.append(loss.item())

            #calculate total correct outputs
            _, pred = torch.max(output, 1)
            if pred.item() == label:
                total_correct += 1
            total_samples += 1
            
            all_preds.append(pred.item())
            all_labels.append(label)

        #log fhe-specific classification metrics
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_loss = test_loss / total_samples
        accuracy = 100 * total_correct / total_samples

        #output and store predictions
        print(f"\n[SUMMARY] Total Encrypted Inference Time: {total_time:.2f}s")
        print(f"[SUMMARY] Average Test Loss: {avg_loss:.6f}")
        print(f"[SUMMARY] Overall Test Accuracy: {int(accuracy)}% ({total_correct}/{total_samples})")
        Ml_Metrics_Evaluation.log_resource_usage(model_type)
        self.fhe_metrics(model_type, encryption_context, total_time, losses, all_preds, all_labels)

    def testing_pipeline(self, plain_model, enc_model, x_test, y_test):
        """
        Full pipeline for encrypted model testing. Generates encryption context,
        extracts parameters from plaintext model, and runs full test suite.
        
        plain_model: trained plaintext model
        enc_model: initialized fhe model used to wrap the trained plaintext model
        x_test: input data
        y_test: ground truth data
        run_type: "Testing" or "Final Model"
        """
        
        print("[INFO] Starting encrypted testing pipeline")
        
        model_type = "encrypted-fhe"
        
        #kernel and stride values from plaintext model used to create CKKS context
        kernel_shape = plain_model.conv1.kernel_size
        stride = plain_model.conv1.stride[0]
        
        #create CKKS encryption context
        encryption_context = ts.context(
            self.scheme_type,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_bit_sizes
        )
        encryption_context.global_scale = pow(2, self.bits_scale)
        encryption_context.generate_galois_keys()
        
        #run encrypted inference test
        self.test_enc_model(model_type, encryption_context, enc_model, x_test, y_test, self.loss_function, kernel_shape, stride)
        
    def fhe_metrics(self, model_type, encryption_context, total_time, test_loss, all_preds, all_labels):
        """
        Send all evaluation results to metric logger module. Includes classification metrics and encryption-related metadata.
        
        model_type(str): "FHE_Model
        encryption_context: initialized CKKS encryption context
        total_time (str): total time it took the model to run
        test_loss (List): computed loss over course of testing the model
        all_preds (np array): all predictions over fhe testing
        all_labels (np array): all ground truth labels
        """
        
        Ml_Metrics_Evaluation.function_time(model_type, total_time)
        Ml_Metrics_Evaluation.clas_accuracy_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.clas_precision_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.clas_recall_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.clas_f1_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.clas_confusion_matrix(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.training_evaluation(model_type, test_loss)
        Ml_Metrics_Evaluation.CKKS_METRICS(
            model_type, self.poly_modulus_degree, self.start_key_generation, self.bits_scale, self.end_decryption_stability, encryption_context 
        )
