import numpy as np

NPZ_FILE = 'C:/Users/Asakura/Desktop/GAME_AI/Project/dreamerv3-torch/logdir/006-fightingice/train_eps/20241213T211512-06e56515b7d04969a0ac7ba7203a8808-116.npz'

npz = np.load(NPZ_FILE)
print(npz.files)
pass