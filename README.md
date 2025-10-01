# rl-durotaxis
Implementation of Durotaxis via Reinforcement Learning


## Dependencies
### Create and Activate a Conda Environment
First, create a fresh conda environment (named `durotaxis`) to avoid any conflicts with existing packages. The supported Python versions of [Deep Graph Library (DGL)](https://www.dgl.ai/pages/start.html) are 3.8, 3.9, 3.10, 3.11, 3.12. 
```bash
conda create -n durotaxis python=3.12.11
```

### Activate the environment
```bash
conda activate durotaxis
```

### Install PyTorch 
You can install PyTorch either using CPU or GPU. The Deep Graph Library (DGL) only supports:
* CUDA 11.8 and 12.1 (Make sure to install first the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local).)
* [PyTorch](https://pytorch.org/get-started/locally/) 2.1.0+

If you wish to use CPU for PyTorch, use the command:
```bash
pip install torch==2.2.1
```

If your computer has NVIDIA CUDA support, use:
```bash
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

### Install Deep Graph Library (DGL)
After PyTorch is installed, you can install DGL. Since 2024.06.27, DGL have stopped providing packages for Windows and MacOS. The latest version of available package is 2.2.1. For DGL installation instructions, go to [DGL page](https://www.dgl.ai/pages/start.html).
For CPU:
```bash
pip install  dgl -f https://data.dgl.ai/wheels/repo.html
```
For GPU:
```bash
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

### Install other packages
```bash
pip install -r requirements.txt
```

### Installation in WSL
Prepare your WSL first.
```bash
wsl.exe --install Ubuntu-24.04
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```
There might be some issues with library compatability. If you are using Windows, I suggest that you create environment in WSL.
```bash
conda create -n durotaxis python=3.12.11
conda activate durotaxis
pip install torch==2.4.0 stable-baselines3 dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install -r requirements.txt
```

# Action space
Investigate if we can use "topology".

# Success termination
If agent reaches the rightmost area.
