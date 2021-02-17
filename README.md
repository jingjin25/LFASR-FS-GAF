# LFASR-FS-GAF
PyTorch implementation of **IEEE TPAMI 2020** paper: "**Deep Coarse-to-fine Dense Light Field Reconstruction with Flexible Sampling and Geometry-aware Fusion**".

[[Paper]](https://ieeexplore.ieee.org/document/9204825)

## Requirements
- Python 3.6
- PyTorch 1.3
- Matlab (for training/test data generation)

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in `LFData`.

## Demo
To reproduce the experimental results presented in the paper, run:

**(Ours (fixed) under task 2x2&rarr;7x7 for synthetic LF data)**
```
python test_pretrained.py --model_dir pretrained_models --save_dir results --arb_sample 0 --angular_out 7 --angular_in 2 --train_dataset HCI --test_dataset HCI --test_path ./LFData/test_HCI.h5 --psv_range 4 --psv_step 50 --input_ind 0 6 42 48 --save_img 1 --crop 1
```
**(Ours (fixed) under task 2x2&rarr;7x7 for Lytro LF data)**
```
python test_pretrained.py --model_dir pretrained_models --save_dir results --arb_sample 0 --angular_out 7 --angular_in 2 --train_dataset SIG --test_dataset 30scenes --test_path ./LFData/test_30scenes.h5 --psv_range 2 --psv_step 50 --input_ind 0 6 42 48 --save_img 1 --crop 1
```
**(Ours (flexible) under task 4&rarr;7x7 for synthetic LF data)**
```
python test_pretrained.py --model_dir pretrained_models --save_dir results --arb_sample 1 --angular_out 7 --angular_in 4 --train_dataset HCI --test_dataset HCI --test_path ./LFData/test_HCI.h5 --psv_range 4 --psv_step 50 --input_ind 16 18 30 32 --save_img 1 --crop 1
```
**(Ours (flexible) under task 4&rarr;7x7 for Lytro LF data)**
```
python test_pretrained.py --model_dir pretrained_models --save_dir results --arb_sample 1 --angular_out 7 --angular_in 4 --train_dataset SIG --test_dataset 30scenes --test_path ./LFData/test_30scenes.h5 --psv_range 2 --psv_step 50 --input_ind 11 15 33 37 --save_img 1 --crop 1
```

## Training
The training code will be released soon.
