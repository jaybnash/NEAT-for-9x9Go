import os
os.environ['JAX_PLATFORMS'] = 'cpu' # This is bc I'm using my GPUs for xpilot, REMOVE THIS
import jax
import jax.numpy as jnp
import pgx
import neat
import numpy as np

# Define the environment initialization and step functions
def create_env():
    # 19x19 is unbearably slow lol
    env = pgx.make("go_9x9")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    return env, init, step

# VERY rough estimator for the "value" of a given move... maybe replace with MCTS? (mctx package cld help)
def estimate_intermediate_score(previous_obs, current_obs, action, legal):
    if not legal[action]:
        return -1  # Technically this should never fire, but if the agent tries to play an illegal move its punished

    # Calculate the difference in the number of pieces on the board
    previous_player_stones = np.sum(previous_obs[:, :, 0])
    previous_opponent_stones = np.sum(previous_obs[:, :, 1])
    current_player_stones = np.sum(current_obs[:, :, 0])
    current_opponent_stones = np.sum(current_obs[:, :, 1])
    territory_gain = (current_player_stones - previous_player_stones) # Delta player pieces
    captured_stones = (previous_opponent_stones - current_opponent_stones) # Delta opp pieces

    # intermediate score is the sum of the deltas in piece number
    intermediate_score = territory_gain + captured_stones
    return intermediate_score

# Function to evaluate individual (this is the main function called by NEAT)
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config) # Create network from NEAT genome
    env, init, step = create_env()
    batch_size = 1 # Batch size is how many games are played per genome evaluation
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    state = init(keys)
    total_reward = 0
    total_intermediate_score = 0

    while not (state.terminated | state.truncated).all(): # The state is actually a batch of states, therefore the .all
        # State observation to numpy, dunno why I need to do this
        observations = np.array(state.observation)
        # observation state is N * N * 17 in size
        legal_actions = np.array(state.legal_action_mask)
        # action space is N * N + 1 in size

        actions = [] # Actions must also be batched since we are batching the states
        intermediate_scores = []  # Track scores for each move
        for obs, legal in zip(observations, legal_actions):
            # This is a dumb way to do this, why tf are we batching just to use zip??
            # The docs imply we don't need to but I didn't have much luck, this works fine
            output = net.activate(obs.flatten().tolist())  # Flatten observations to NEAT input
            action = np.argmax(output * legal)  # Mask illegal actions
            actions.append(action)  # Append action to batch actions

        actions = np.array(actions)  # ¯\_(ツ)_/¯
        previous_observations = np.copy(observations)  # Save state observations for later eval
        state = step(state, actions)  # state.rewards with shape (batch_size, 2)
        total_reward += jnp.sum(state.rewards)  # Sum state rewards to total rewards
        # Note that state rewards are only either 1, -1, or 0, win/lose/game-ongoing

        # Estimate intermediate scores for each move
        current_observations = np.array(state.observation)  # Use resulting state of NEAT action for eval
        for prev_obs, curr_obs, action, legal in zip(previous_observations, current_observations, actions, legal_actions):
            # Again... I don't think we need zips but this works ig
            intermediate_score = estimate_intermediate_score(prev_obs, curr_obs, action, legal)  # Estimates score per move
            intermediate_scores.append(intermediate_score)  # Appends estimated score each move

        total_intermediate_score += np.sum(intermediate_scores)  # Sum intermediate scores

    # Combine total reward and intermediate scores for final fitness
    final_score = int((total_reward*100) + total_intermediate_score) # A weight is applied to total_reward as that reflects wins/losses
    return final_score

# Define the NEAT configuration
def run_neat(config_file):
    # Literally everything abt the NEAT algorithm is default, except the input/outputs sizes to fit the go space
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)  # Makes a population

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))  # Records a checkpoint every X generations

    winner = p.run(eval_genomes, 50)  # Runs NEAT for X generations, using eval_genomes as the fitness function

    print('\nBest genome:\n{!s}'.format(winner))  # Prints the best genome at end of training... not super helpful

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

if __name__ == '__main__':
    run_neat('neat_config.txt')  # Options for NEAT are stored in a .txt file for some reason
