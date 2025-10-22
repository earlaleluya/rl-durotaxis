# rl-durotaxis
Implementation of Durotaxis via Reinforcement Learning


# Dependencies
Due to compatability issues, it is best to prepare the environment in Ubuntu OS. Optionally, you can use Windows Subsystem for Linux (WSL). To install, open PowerShell and run:
```bash
wsl.exe --install -d Ubuntu-24.04
```

Close and restart the terminal. Open Ubuntu Terminal. Then, install Miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Create a new conda environment:
```bash
conda create -n durotaxis python=3.12.11
conda activate durotaxis
pip install torch==2.4.0 torchvision stable-baselines3 dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

Clone the files to your preferred directory. Then in Ubuntu terminal,
```bash
cd /path/to/your/folder/rl-durotaxis
pip install -r requirements.txt
```


# Training the Agent

The configuration parameters are saved in [config.yaml](config.yaml). You can change the values for your preference. Then, train the agent.

```bash
python train.py
```
