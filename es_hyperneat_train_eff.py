import os
from itertools import combinations

os.environ['JAX_PLATFORMS'] = 'cpu'  # Remove this if not needed
import jax
import jax.numpy as jnp
import pgx
import neat
import numpy as np
import random
import pickle
from NEATOverwrites import Reproduction

# Import PUREPLES for ES-HyperNEAT implementation
from pureples.shared import substrate
from pureples.es_hyperneat import es_hyperneat

global generation_num
generation_num = 8499

# Define input coordinates for the 17 feature planes
input_coordinates = []
for y in range(9):
    for x in range(9):
        x_coord = (x / 8.0) * 2 - 1
        y_coord = (y / 8.0) * 2 - 1
        input_coordinates.append((x_coord, y_coord))

# Output coordinates remain the same
output_coordinates = input_coordinates.copy()
# Add the "pass" action coordinate
output_coordinates.append((0.0, 0.0))

sub = substrate.Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT parameters
params = {"initial_depth": 2,
            "max_depth": 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 5.0,
            "activation": "relu"}

# Define the environment initialization and step functions
def create_env():
    env = pgx.make("go_9x9")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    return env, init, step


def parse_observation(obs):
    # obs is of shape (9, 9, 17)

    # Current player's stones at the current board (boolean array)
    current_player_stones = obs[:, :, 0]

    # Opponent's stones at the current board (boolean array)
    opponent_player_stones = obs[:, :, 1]

    # Initialize simplified observation array
    simplified_obs = np.zeros((9, 9), dtype=int)

    # Set positions with current player's stones to 1
    simplified_obs[current_player_stones] = 1

    # Set positions with opponent's stones to -1
    simplified_obs[opponent_player_stones] = -1

    # Flatten to create a 1D array of 81 elements
    simplified_obs_flat = simplified_obs.flatten()
    return simplified_obs_flat


# Function to evaluate genomes in batches
def eval_genomes(genomes, config):
    global generation_num
    if generation_num == 0:
        for genome_id, genome in genomes:
            genome.bias = 1
            
    for genome_id, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        es_net = es_hyperneat.ESNetwork(sub, cppn, params)
        genome.phenotype_net = es_net.create_phenotype_network()

    new_bias = False
    save = False
    test_best = False

    if generation_num % 50 == 0 and generation_num != 0:
        new_bias = True
        test_best = True

    indices = list(range(len(genomes)))
    random.shuffle(indices)
    for genome_id, genome in genomes:
        genome.fitness = 0

    batch_size = 100  # Adjust the batch size as needed

    if new_bias:
        batch_size = 4950
        genome_pairs = list(combinations(indices, 2))
        random.shuffle(genome_pairs)
        for i in range(0, len(genome_pairs), batch_size):
            genome_pairs_batch_indices = genome_pairs[i:i + batch_size]
            genome_pairs_batch = []
            for idx1, idx2 in genome_pairs_batch_indices:
                genome_id_1, genome_1 = genomes[idx1]
                genome_id_2, genome_2 = genomes[idx2]
                genome_pairs_batch.append((genome_1, genome_2))
            fit_1_list, fit_2_list = eval_genome_vs_genome_batch(genome_pairs_batch, config)
            for (genome_1, genome_2), fit_1, fit_2 in zip(genome_pairs_batch, fit_1_list, fit_2_list):
                genome_1.fitness += float(fit_1)
                genome_2.fitness += float(fit_2)
        for genome_id, genome in genomes:
            genome.bias = genome.fitness
    else:
        for _ in range(5):
            random.shuffle(indices)
            for i in range(0, len(indices) - 1, 2 * batch_size):
                batch_indices = indices[i:i + 2 * batch_size]
                genome_pairs_batch = []
                for j in range(0, len(batch_indices) - 1, 2):
                    idx1, idx2 = batch_indices[j], batch_indices[j + 1]
                    genome_id_1, genome_1 = genomes[idx1]
                    genome_id_2, genome_2 = genomes[idx2]
                    genome_pairs_batch.append((genome_1, genome_2))
                fit_1_list, fit_2_list = eval_genome_vs_genome_batch(genome_pairs_batch, config)
                for (genome_1, genome_2), fit_1, fit_2 in zip(genome_pairs_batch, fit_1_list, fit_2_list):
                    genome_1.fitness += float(fit_1 * genome_1.bias)
                    genome_2.fitness += float(fit_2 * genome_2.bias)
                    if save:
                        save = False

    if test_best:
        best_genome = None
        for genome_id, genome in genomes:
            if best_genome is None or genome.fitness > best_genome.fitness:
                best_genome = genome
        eval_genome_vs_baseline(best_genome, config)

    generation_num += 1

def eval_genome_vs_baseline(genome1, config):
    # Create the CPPN network from the genome
    cppn = neat.nn.FeedForwardNetwork.create(genome1, config)
    # Create the ESNetwork with the substrate, CPPN, and parameters
    es_net = es_hyperneat.ESNetwork(sub, cppn, params)
    # Create the phenotype network
    net1 = es_net.create_phenotype_network()

    model_id = 'go_9x9_v0'
    model = pgx.make_baseline_model(model_id)
    env, init, step = create_env()
    batch_size = 1
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    state = init(keys)
    genome1_reward = 0
    baseline_reward = 0
    states = [state]
    while not (state.terminated | state.truncated).all():
        observations = np.array(state.observation)
        legal_actions = np.array(state.legal_action_mask)
        output = net1.activate(observations.flatten().tolist())
        action_A = np.argmax(output * legal_actions)
        logits, value = model(state.observation)
        action_B = logits.argmax(axis=-1)
        actions = jnp.where(state.current_player == 0, action_A, action_B)
        state = step(state, actions)
        states.append(state)
        rewards = state.rewards
        genome1_reward += jnp.sum(rewards[:, 0])
        baseline_reward += jnp.sum(rewards[:, 1])
    global generation_num
    with open(f"./es_data/trial/{generation_num}_genome.pkl", 'wb') as file:
        pickle.dump(genome1, file)
    if genome1_reward >= baseline_reward:
        pgx.save_svg_animation(states, f"./es_data/trial/{generation_num}_trial_WIN.svg", frame_duration_seconds=0.2)
    else:
        pgx.save_svg_animation(states, f"./es_data/trial/{generation_num}_trial_LOSS.svg", frame_duration_seconds=0.2)


def eval_genome_vs_genome_batch(genome_pairs_batch, config):
    batch_size = len(genome_pairs_batch)
    phenotype_net1_list = []
    phenotype_net2_list = []

    for genome1, genome2 in genome_pairs_batch:
        phenotype_net1_list.append(genome1.phenotype_net)
        phenotype_net2_list.append(genome2.phenotype_net)

    env, init, step = create_env()
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    state = init(keys)
    genome1_rewards = np.zeros(batch_size)
    genome2_rewards = np.zeros(batch_size)

    while not (state.terminated | state.truncated).all():
        observations = np.array(state.observation)
        legal_actions = np.array(state.legal_action_mask)
        actions = []

        for i in range(batch_size):
            obs = observations[i]
            legal = legal_actions[i]
            obs_simplified = parse_observation(obs)  # Use the simplified observation
            if state.current_player[i] == 0:
                output = phenotype_net1_list[i].activate(obs_simplified.tolist())
            else:
                output = phenotype_net2_list[i].activate(obs_simplified.tolist())
            output = np.array(output)
            action = np.argmax(output * legal)
            actions.append(action)

        actions = np.array(actions)
        state = step(state, actions)
        rewards = np.array(state.rewards)
        genome1_rewards += rewards[:, 0]
        genome2_rewards += rewards[:, 1]

    fit_1_list = []
    fit_2_list = []
    for i in range(batch_size):
        if genome1_rewards[i] > genome2_rewards[i]:
            fit_1_list.append(1)
            fit_2_list.append(0)
        elif genome2_rewards[i] > genome1_rewards[i]:
            fit_1_list.append(0)
            fit_2_list.append(1)
        else:
            fit_1_list.append(0.5)
            fit_2_list.append(0.5)

    return fit_1_list, fit_2_list

# Define the NEAT configuration for ES-HyperNEAT
def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, Reproduction.OverwriteReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    try:
        p = neat.Checkpointer.restore_checkpoint('es_data/restore_point.pkl')
    except Exception as e:
        print(e)
        p = neat.Population(config)  # Create a new population

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=50, time_interval_seconds=None,
                                     filename_prefix="./es_data/checkpoints/"))
    winner = p.run(eval_genomes, 10000)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    run_neat('es_config.txt')
