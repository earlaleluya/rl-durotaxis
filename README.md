# rl-durotaxis
Implementation of Durotaxis via Reinforcement Learning


# Dependencies
Due to compatability issues, it is best to prepare the environment in Ubuntu OS. Optionally, you can use Windows Subsystem for Linux (WSL).
```bash
conda create -n durotaxis python=3.12.11
conda activate durotaxis
pip uninstall torch stable-baselines3 dlg -y
pip install torch stable-baselines3 dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html


pip install torch==2.4.0 stable-baselines3==2.7.0 dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

Clone the files to your preferred directory. Then in terminal,
```bash
cd /path/to/your/folder/rl-durotaxis
pip install -r requirements.txt
```


# Training the Agent

The configuration parameters are saved in [config.yaml](config.yaml). You can change the values for your preference. Then, train the agent.

```bash
python train.py
```


## Curriculum Learning & Enhanced Success Criteria 🎓
3-Phase Training System:
* Phase 1 (0-300 episodes): Easy mode with reduced complexity (30 max nodes, 3 starting nodes, 50% penalty reduction)
* Phase 2 (300-600 episodes): Medium mode with gradual increase (40 max nodes, 2 starting nodes, 75% penalty reduction)
* Phase 3 (600+ episodes): Full complexity with standard parameters

Multiple Success Criteria:
* Survival success: Maintain nodes for 10+ steps
* Reward success: Achieve reasonable reward (-20+ threshold)
* Growth success: Maintain connectivity (2+ nodes)
* Exploration success: Complete long episodes (15+ steps)
