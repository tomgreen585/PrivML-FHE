import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import tenseal as ts
from io import BytesIO
from fhe_br_evaluation import Ml_Metrics_Evaluation, encrypted_predicted_images
from fhe_br_config import (SCHEME_TYPE, ENCRYPTED_LOSS_FUNCTION, BITS_SCALE, POLY_MODULUS_DEGREE, ST_KEY_GENERATION, END_DECRYPTION_STABILITY, SAMPLE_OUTPUT_COUNT)

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
    sample_output_count: number of samples to test
    """
   
    def __init__(self):
        #load encryption and model parameters from config file
        self.scheme_type = SCHEME_TYPE
        self.loss_function = ENCRYPTED_LOSS_FUNCTION
        self.bits_scale = BITS_SCALE
        self.poly_modulus_degree = POLY_MODULUS_DEGREE
        self.start_key_generation = ST_KEY_GENERATION
        self.end_decryption_stability = END_DECRYPTION_STABILITY
        self.sample_count = SAMPLE_OUTPUT_COUNT
        self.testing_completed = False

    def test_enc_model(self, enc_model, x_test_enc, y_test, kernel_shape, stride, encryption_context, run_type, model_type):
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

        all_preds = []
        all_labels = []
        losses = []

        total_start_time = time.time()
        
        for idx in range(self.sample_count):
            #log CPU/memory usage
            Ml_Metrics_Evaluation.log_resource_usage(model_type)
            
            sample = x_test_enc[idx]
            label = y_test[idx]

            #convert to numpy
            if isinstance(sample, torch.Tensor):
                sample = sample.detach().cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy()
            
            #convert to grayscale if needed
            if sample.shape[0] == 3:
                sample = np.mean(sample, axis=0, keepdims=True)
            if sample.shape[-1] == 1:
                sample = np.transpose(sample, (2, 0, 1))
            
            C, H, W = sample.shape
            enc_channels = []
            windows_nb = None
            #encode each channel using im2col and CKKS
            for c in range(C):
                channel = sample[c] / 255.0
                ch = np.pad(channel, ((1, 1), (1, 1)), mode="constant")
                enc_ch, windows_nb = ts.im2col_encoding(
                    encryption_context, 
                    ch.tolist(),
                    kernel_shape[0], kernel_shape[1], stride
                )
                enc_channels.append(enc_ch)

            #run inference on encrypted model
            pred = enc_model(enc_channels, windows_nb)
            pred_np = np.array(pred.decrypt()).reshape((1, H, W)) * 255.0
            
            #decrypt and reshape prediction
            output_tensor = torch.tensor(np.array(pred.decrypt())).view(1, -1)
            label_tensor = torch.tensor(label).view(1, -1)
            
            #compute loss
            loss = self.loss_function(output_tensor, label_tensor)
            losses.append(loss.item())
            pred_np = np.array(pred.decrypt()).reshape((1, H, W)) * 255.0
            all_preds.append(pred_np)
            all_labels.append(label)
        
        #reshape for metric computation    
        y_true = np.array(all_labels).reshape(len(all_labels), -1)
        y_pred = np.array(all_preds).reshape(len(all_preds), -1)

        #log fhe-specific regression metrics
        end_time = time.time()
        total_time = end_time - total_start_time
        self.fhe_metrics(model_type, encryption_context, total_time, losses, y_pred, y_true)
        
        #display and store predictions
        self.visualize_predictions(x_test_enc, y_test, all_preds, run_type, self.sample_count)
        self.testing_completed = True

    def fhe_metrics(self, model_type, encryption_context, total_time, test_loss, all_preds, all_labels):
        """
        Send all evaluation results to metric logger module. Includes regression metrics and encryption-related metadata.
        
        model_type(str): "FHE_Model
        encryption_context: initialized CKKS encryption context
        total_time (str): total time it took the model to run
        test_loss (List): computed loss over course of testing the model
        all_preds (np array): all predictions over fhe testing
        all_labels (np array): all ground truth labels
        """
        Ml_Metrics_Evaluation.function_time(model_type, total_time)
        Ml_Metrics_Evaluation.reg_mean_squared_error(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.reg_mean_absolute_error(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.reg_mean_absolute_percentage_error(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.reg_r2_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.reg_explained_variance_score(model_type, all_labels, all_preds)
        Ml_Metrics_Evaluation.training_evaluation(model_type, test_loss)
        Ml_Metrics_Evaluation.CKKS_METRICS(
            model_type, self.poly_modulus_degree, self.start_key_generation,
            self.bits_scale, self.end_decryption_stability, encryption_context
        )

    def visualize_predictions(self, x_test, y_test, all_preds, run_type, num_samples):
        """
        Creates matplotlib visualisation of encrypted model predictions.
        Displays (or stores) side-by-side input, ground truth, and predicted output.
        
        x_test: input image
        y_test: ground truth output
        all_preds: all predicted outputs from FHE model
        run_type: "Testing" or "Final Model"
        num_samples: number of samples to visualize and generate plots
        """
        
        print("[INFO] Visualizing Predictions...")

        num_samples = min(num_samples, len(all_preds))
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4))

        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_samples):
            sample = x_test[i].squeeze()
            label = y_test[i].squeeze()
            pred = all_preds[i].squeeze()

            axes[i, 0].imshow(sample, cmap="gray")
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(label, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Predicted Output")
            axes[i, 2].axis("off")

        fig.suptitle(f"Encrypted Inference Predictions ({run_type})", fontsize=16)
        title = "Encrypted Predictions"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encrypted_predicted_images.append(("Encrypted Predicted Image", buf))
        if run_type == "Testing":
            plt.show()
        else:
            plt.close()
            
    def testing_pipeline(self, plain_model, enc_model, x_test, y_test, run_type):
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
            coeff_mod_bit_sizes=[
                self.start_key_generation, 
                self.bits_scale, self.bits_scale, self.bits_scale, self.bits_scale, 
                self.end_decryption_stability
            ]
        )
        encryption_context.global_scale = pow(2, self.bits_scale)
        encryption_context.generate_galois_keys()

        #run encrypted inference test
        self.test_enc_model(enc_model, x_test, y_test, kernel_shape, stride, encryption_context, run_type, model_type)
