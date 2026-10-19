# NumPy vs. PyTorch comparison

## Results

| Model | Device | Train images | Epochs | Test images | Train time | Train img/s | Test time | Test ms/img | Train speed-up | Test speed-up | Test accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NumPyCNN | cpu (AMD Ryzen 7 9800X3D 8-Core Processor) | 18727 | 10 | 4682 | 26.31 s | 7,118 | 493.4 ms | 0.105 | 1.0x | 1.0x | 54.25% |

Speed-ups are relative to NumPyCNN (cpu (AMD Ryzen 7 9800X3D 8-Core Processor)).
`Test time` is the total for all test images; `Test ms/img` is the per-image average.
Only the NumPy baseline is available. The 1.0x values compare the baseline with
itself; GPU speedups require a real PyTorch GPU result.

## Run configuration

| Model | Device | Machine | Split fingerprint | Seed | Image size | Other settings |
|---|---|---|---|---|---|---|
| NumPyCNN | cpu (AMD Ryzen 7 9800X3D 8-Core Processor) | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 / Python 3.12.13 | 5deb0f27f6c5 | 3 | 64x64 | batch_size=1, convolution=NumPy vectorized einsum, data_seed=3, inference_batch_size=64, lr=0.001, max_test_per_class=0, max_train_per_class=0, test_block_size=1000, trained_layers=fully connected only |

## Findings

- The run used the data split with fingerprint `5deb0f27f6c5`.
