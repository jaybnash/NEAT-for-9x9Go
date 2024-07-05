Script that trains a NEAT algorithm to play 9x9 Go. The NEAT parameters are controlled by the text file.

neat_train trains NEAT agents to beat other NEAT agents in the population. Each agent faces a random other agent in the population every generation and fitness is determined by the winner (1, 0, or 0.5 for tie). Every 200 generations fitness biasing is preformed by trialing every agent against every other agent.

Additionally, every 200 generations the best agent created by NEAT is pitted against the baseline 9x9 go model provided by pgx. This baseline model uses the Grumbel AlphaZero algorithm but has had its training stopped after it reached 1000 Elo. The authors of pgx and this model specifically state that this model is not meant to be a oracle or state-of-the-art model, but a general baseline of a good Go Player.

SVG animations are saved of games between NEAT agents every 50 generations while SVG animations are always saved of trials between the best NEAT agent and the baseline model. For SVG animations of the NEAT model facing the baseline model, the win/loss state of the NEAT player is appended to the filename.
