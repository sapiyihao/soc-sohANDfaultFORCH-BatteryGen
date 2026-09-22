# 训练与检查环境

本文件记录打包时项目指定 Python 环境的实际状态。

- Python 路径：`D:\enviroment\anaconda\envs\py312\python.exe`
- 操作系统：Windows 10 10.0.19045
- Python：3.12.13（Anaconda）
- PyTorch：2.5.1+cu121
- PyTorch CUDA build：12.1
- CUDA 可用：是
- GPU：NVIDIA GeForce RTX 3070 Laptop GPU
- 显存：8192 MiB
- NVIDIA 驱动：592.82
- NumPy：2.4.6
- pandas：3.0.3
- Matplotlib：3.10.9
- scikit-learn：1.8.0

运行前可执行：

```powershell
& 'D:\enviroment\anaconda\envs\py312\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

