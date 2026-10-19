# NumPy vs. PyTorch comparison

## Results

| Model | Device | Train images | Epochs | Test images | Train time | Train img/s | Test time | Test ms/img | Compute ms/img | Train speed-up | Test speed-up | Test accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NumPyCNN | cpu (AMD Ryzen 7 9800X3D 8-Core Processor) | 18727 | 10 | 4682 | 26.31 s | 7,118 | 493.4 ms | 0.105 | 0.095 | 1.0x | 1.0x | 54.25% |
| CatDogCNN | cuda (NVIDIA GeForce RTX 5080) | 18727 | 10 | 4682 | 5.65 s | 33,171 | 158.8 ms | 0.034 | 0.008 | 4.7x | 3.1x | 78.68% |

Speed-ups are relative to NumPyCNN (cpu (AMD Ryzen 7 9800X3D 8-Core Processor)). `Test time` is the total for all test images; `Test ms/img` is the per-image average; `Compute ms/img` counts only the model's forward pass.

## Run configuration

| Model | Device | Machine | Trained parameters | Split fingerprint | Seed | Image size | Other settings |
|---|---|---|---|---|---|---|---|
| NumPyCNN | cpu (AMD Ryzen 7 9800X3D 8-Core Processor) | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 / Python 3.12.13 | 962 | 5deb0f27f6c5 | 3 | 64x64 | batch_size=1, convolution=NumPy vectorized einsum, data_seed=3, inference_batch_size=64, lr=0.001, max_test_per_class=0, max_train_per_class=0, test_block_size=1000, trained_layers=fully connected only |
| CatDogCNN | cuda (NVIDIA GeForce RTX 5080) | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 / Python 3.10.20 | 529,505 | 5deb0f27f6c5 | 3 | 64x64 | batch_size=64, cpu_threads=4, lr=0.001, max_test_per_class=0, max_train_per_class=0, num_workers=0, test_block_size=1000 |

## Findings

- Both runs used the data split with fingerprint `5deb0f27f6c5`.
- CatDogCNN on cuda (NVIDIA GeForce RTX 5080) trains 4.7x faster and tests 3.1x faster per image than NumPyCNN, with test accuracy 78.68% vs 54.25% (+24.43 percentage points).
- Counting only model computation (no batching or metrics), CatDogCNN tests 12.2x faster than NumPyCNN.
- The models are not the same network: CatDogCNN trains 529,505 parameters and NumPyCNN 962, so the speed-up reflects both the hardware/framework and the different amount of work per image, and the accuracy gap mainly reflects the model.
