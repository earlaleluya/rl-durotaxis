# rl-durotaxis
Implementation of Durotaxis via Reinforcement Learning


## Dependencies
### Installation in WSL
In PowerShell, install WSL with Ubuntu-24.04 first in your Windows OS.
```bash
wsl.exe --install Ubuntu-24.04
```

Close and restart the terminal. Open the Ubuntu terminal, not PowerShell. The next command downloads Miniconda.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Install the Miniconda using bash command below.
```bash
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Next, setup the Conda environment. 
```bash
conda create -n durotaxis python=3.12.11
```

Activate the environment.
```bash
conda activate durotaxis
```

Install PyTorch, Stable-baselines3, and Deep Graph Library (DGL) packages in one go.
```bash
pip install torch==2.4.0 stable-baselines3 dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

Download the file using git.
```bash
git clone https://github.com/earlaleluya/rl-durotaxis.git
```

Install other packages.
```bash
pip install -r requirements.txt
```

# Action space
Investigate if we can use "topology".

# Success termination
If agent reaches the rightmost area.
