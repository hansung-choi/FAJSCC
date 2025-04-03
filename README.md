# FAJSCC

Implementations of main experiments for the paper "Feature Importance-Aware Deep Joint Source-Channel Coding for Computationally Efficient Image Transmission"

## Requirements
1. python 3.8.8
2. pytorch 2.0.1
3. cuda 11.1.1
4. numpy 1.24.4
5. hydra 1.1

## Experiment code manual

### Arguments for terminal execution
1. **chan_type**: The type of communication channel, which can be either **"AWGN" or "Rayleigh"**.
2. **rcpp**: The reciprocal of **cpp** (channel usage per RGB pixels). It can take one of the following discrete values: **12, 16, or 24**.
3. **SNR_info**: The channel SNR value, which can be one of **1, 4, 7, or 10** dB.
4. **performance_metric**: The performance metric to be maximized, which can be either **"PSNR" or "SSIM"**.
5. **data_info**: The dataset name (possible value: **"DIV2K"**).
6. **model_name**: The model name, which can be one of the following: **"smallConvJSCC", "baseConvJSCC", "smallResJSCC", "baseResJSCC", "smallSwinJSCC", "baseSwinJSCC", "smallFAwoSIJSCC", "baseFAwoSIJSCC", "smallFAJSCC", or "baseFAJSCC"**.


### Example of training a model where the training SNR matches the test SNR.

    python3 main_train.py rcpp=12 chan_type=AWGN performance_metric=PSNR SNR_info=4 model_name=smallConvJSCC data_info=DIV2K


### Example of training a model with a randomly assigned SNR for each data batch.
**Only "smallFAJSCC" or "baseFAJSCC" models are allowed.**

    python3 main_train.py rcpp=12 chan_type=AWGN performance_metric=PSNR SNR_info="random" model_name=smallFAJSCC data_info=DIV2K


### Example of experimental results for "Architecture Efficiency".
**You can obtain test results for other settings by simply modifying the SNR or rcpp values in the main_total_evalGM.py file.**

    python3 main_total_evalGM.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "PSNR and SSIM Results".
**You can obtain test results for other settings by simply modifying the rcpp values in the main_total_eval.py file**

    python3 main_total_eval.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "Computation Resource Adjustment" and "Complexity of Encoder and Decoder".
**You can obtain test results for other settings by simply modifying the SNR or rcpp values in the main_total_evalratio.py file**

    python3 main_total_evalratio.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "Visual Inspection".

    python3 main_model_visualize.py  SNR_info=1 rcpp=12 chan_type="AWGN" performance_metric="PSNR" model_name=baseResJSCC data_info=DIV2K



