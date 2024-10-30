import numpy as np

# Assuming playerStats is already defined
playerStats = matchPlayersStats.to_numpy().reshape(-1, 22, matchPlayersStats.width)

# Save the numpy array to a file
np.save('playerstats.npy', playerStats)

print("playerStats saved to playerstats.npy")
