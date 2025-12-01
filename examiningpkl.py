import pickle

with open("checkpoints/nfsp/final_60.pkl", "rb") as f:
    checkpoint = pickle.load(f)

# Inspect the contents
print(checkpoint.keys())  # Check the top-level keys in the checkpoint

# dict_keys(['episode', 'config', 'metrics_history', 'agent_params'])

print("Episode:", checkpoint['episode'])
print("Config:", checkpoint['config'])
print("Metrics history keys:", checkpoint['metrics_history'])
print("Number of agents' params saved:", len(checkpoint['agent_params']))
for i, agent_param in enumerate(checkpoint['agent_params']):
    print(f"Agent {i} params keys:", agent_param.keys())

#result:
# Config: {'game_name': 'truco', 'num_train_episodes': 10000, 'eval_every': 2000, 'hidden_layers_sizes': (128, 128), 'replay_buffer_capacity': 100, 'reservoir_buffer_capacity': 1000, 'anticipatory_param': 0.1, 'epsilon_start': 0.06, 'epsilon_end': 0.001, 'epsilon_decay_duration': None, 'checkpoint_dir': PosixPath('checkpoints/nfsp'), 'checkpoint_freq': 50000, 'log_freq': 1000, 'seed': 42}
# Metrics history keys: [{'episode': 2000, 'losses': [(Array(3.2885275, dtype=float32), None), (Array(3.4246082, dtype=float32), None)], 'win_rate_vs_random': 0.48, 'games_vs_random': 50, 'win_rate_vs_ismcts_100': 0.22, 'games_vs_ismcts_100': 50}, {'episode': 4000, 'losses': [(Array(3.0095701, dtype=float32), None), (Array(3.1195822, dtype=float32), None)], 'win_rate_vs_random': 0.5, 'games_vs_random': 50, 'win_rate_vs_ismcts_100': 0.22, 'games_vs_ismcts_100': 50}, {'episode': 6000, 'losses': [(Array(2.7580204, dtype=float32), None), (Array(2.746947, dtype=float32), None)], 'win_rate_vs_random': 0.5, 'games_vs_random': 50, 'win_rate_vs_ismcts_100': 0.36, 'games_vs_ismcts_100': 50}, {'episode': 8000, 'losses': [(Array(2.6918802, dtype=float32), None), (Array(2.6294208, dtype=float32), None)], 'win_rate_vs_random': 0.54, 'games_vs_random': 50, 'win_rate_vs_ismcts_100': 0.36, 'games_vs_ismcts_100': 50}, {'episode': 10000, 'losses': [(Array(2.4938087, dtype=float32), None), (Array(2.1808863, dtype=float32), None)], 'win_rate_vs_random': 0.6, 'games_vs_random': 50, 'win_rate_vs_ismcts_100': 0.38, 'games_vs_ismcts_100': 50}]
# Number of agents' params saved: 2
# Agent 0 params keys: dict_keys(['params_avg_network', 'params_q_network'])
# Agent 1 params keys: dict_keys(['params_avg_network', 'params_q_network'])

