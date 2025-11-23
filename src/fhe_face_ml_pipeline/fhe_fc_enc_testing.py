import torch
import torch.nn as nn
import time
import tenseal as ts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from fhe_fc_evaluation import Ml_Metrics_Evaluation, actual_prediction_encrypted, actual_vs_real_prediction_encrypted
from fhe_fc_config import (
    SCHEME_TYPE, ENCRYPTED_LOSS_FUNCTION, 
    BITS_SCALE, POLY_MODULUS_DEGREE, 
    ST_KEY_GENERATION, END_DECRYPTION_STABILITY, SAMPLE_OUTPUT_COUNT
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

    def test_enc_model(self, model_type, encryption_context, enc_model, x_test_enc, y_test, kernel_shape, stride, run_type):
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
        print(f"[INFO] Total samples to process {len(x_test_enc)}")

        for idx in range(self.sample_count):
            
            if idx % 50 == 0:
                print(f"[INFO] Processing sample {idx}/{len(x_test_enc)}")
                #log CPU/memory usage
                Ml_Metrics_Evaluation.log_resource_usage(model_type)

            sample = x_test_enc[idx]
            label = y_test[idx].squeeze().tolist()

            #convert to numpy
            if isinstance(sample, torch.Tensor):
                sample = sample.detach().cpu().numpy()
            if sample.ndim == 3 and sample.shape[-1] == 3:
                sample = np.transpose(sample, (2, 0, 1))

            channels = []
            windows_nb = None
            #encode each channel using im2col and CKKS
            for c in range(sample.shape[0]):
                ch = sample[c]
                x_enc_channel, windows_nb = ts.im2col_encoding(
                    encryption_context,
                    ch.tolist(),
                    kernel_shape[0], kernel_shape[1], stride
                )
                channels.append(x_enc_channel)

            #run inference on encrypted model
            enc_output = enc_model(channels, windows_nb)

            #decrypt and reshape prediction
            output = enc_output.decrypt()
            output_tensor = torch.tensor(output).view(1, -1)
            label_tensor = torch.tensor(label).view(1, -1)
            
            #compute loss
            loss = self.loss_function(output_tensor, label_tensor)
            losses.append(loss.item())
            all_preds.append(output_tensor.view(-1).tolist())
            all_labels.append(label_tensor.view(-1).tolist())

            print("###########################################################################")
            print(f"Sample {idx+1}:")
            print("Predicted BBox:", all_preds[-1])
            print("Ground Truth BBox:", all_labels[-1])

        #log fhe-specific regression metrics
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        self.fhe_metrics(model_type, encryption_context, total_time, losses, all_preds, all_labels)

        #display and store predictions
        for i in range(self.sample_count):
            print("[INFO] Generating predictions")
            img = x_test_enc[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            self.visualize_predictions(
                image=img,
                pred_box=torch.tensor(all_preds[i]),
                gt_box=torch.tensor(all_labels[i]),
                run_type=None,
                p_type="JPred"
            )

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
            model_type, self.poly_modulus_degree, self.start_key_generation, self.bits_scale, self.end_decryption_stability, encryption_context 
        )

    def visualize_predictions(self, image, pred_box, gt_box, run_type, p_type):
        """
        Creates bounding box prediction visualisations using Matplotlib

        Args:
            image: original input image
            pred_box: predicted output box from model
            gt_box: ground truth box
            run_type (str): "Training" or "Final_Model"
            p_type (str): type of image ("JPred" = pred only, else = pred vs ground truth).
        """
        
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        h, w = image.shape[:2]
        px, py, pw, ph = pred_box
        px, py, pw, ph = px * w, py * h, pw * w, ph * h
        top_left_pred = (px - pw / 2, py - ph / 2)
        pred_rect = patches.Rectangle(top_left_pred, pw, ph, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(pred_rect)

        if gt_box is not None:
            gx, gy, gw, gh = gt_box
            gx, gy, gw, gh = gx * w, gy * h, gw * w, gh * h
            top_left_gt = (gx - gw / 2, gy - gh / 2)
            gt_rect = patches.Rectangle(top_left_gt, gw, gh, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(gt_rect)

        title = "Green: Predicted | Red: Ground Truth"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        if p_type == "JPred":
            actual_prediction_encrypted.append(("Predicted Image", buf))
        else:
            actual_vs_real_prediction_encrypted.append(("Actual (Red) vs Predicted (Green)", buf))
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
            coeff_mod_bit_sizes=[self.start_key_generation, self.bits_scale, self.bits_scale, self.bits_scale, self.bits_scale, self.bits_scale, self.bits_scale, self.end_decryption_stability]
        )
        encryption_context.global_scale = pow(2, self.bits_scale)
        encryption_context.generate_galois_keys()

        #run encrypted inference test
        self.test_enc_model(model_type, encryption_context, enc_model, x_test, y_test, kernel_shape, stride, run_type)