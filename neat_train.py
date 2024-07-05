import os
from itertools import combinations

os.environ['JAX_PLATFORMS'] = 'cpu' # This is bc I'm using my GPUs for xpilot, REMOVE THIS
import jax
import jax.numpy as jnp
import pgx
import neat
import numpy as np
import random
import pickle
from NEATOverwrites import Reproduction

global generation_num
generation_num = 0

# Define the environment initialization and step functions
def create_env():
    # 19x19 is unbearably slow lol
    env = pgx.make("go_9x9")
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    return env, init, step

# Function to evaluate individual (this is the main function called by NEAT)
def eval_genomes(genomes, config):
    global generation_num
    if generation_num == 0:
        for genome_id, genome in genomes:
            genome.bias = 1

    new_bias = False
    save = True
    test_best = True

    #if generation_num % 50 == 0 and generation_num > 0:
        #new_bias = True
    #elif (generation_num+1) % 50 == 0 and generation_num > 0:
        #save = True
        #test_best = True

    indices = list(range(len(genomes)))  # Create a list of indices
    random.shuffle(indices)  # Shuffle the indices to ensure random pairing
    for genome_id, genome in genomes:
        genome.fitness = 0
    if new_bias:
        genome_pairs = list(combinations(indices, 2))  # Use shuffled indices for combinations
        for idx1, idx2 in genome_pairs:
            genome_id_1, genome_1 = genomes[idx1]
            genome_id_2, genome_2 = genomes[idx2]
            fit_1, fit_2 = eval_genome_vs_genome(genome_1, genome_2, config, save)
            genome_1.fitness += float(fit_1)
            genome_2.fitness += float(fit_2)
        for genome_id, genome in genomes:
            genome.bias = genome.fitness
    else:
        for i in range(3):
            for i in range(0, len(genomes), 2):
                idx1, idx2 = indices[i], indices[i + 1]  # Use shuffled indices for pairing
                genome_id_1, genome_1 = genomes[idx1]
                genome_id_2, genome_2 = genomes[idx2]
                fit_1, fit_2 = eval_genome_vs_genome(genome_1, genome_2, config, save)
                genome_1.fitness += float(fit_1 * genome_1.bias)
                genome_2.fitness += float(fit_2 * genome_2.bias)
                if save:
                    save = False
            random.shuffle(indices)

    if test_best:
        best_genome = None
        for genome_id, genome in genomes:
            if best_genome is None or genome.fitness > best_genome.fitness:
                best_genome = genome
        eval_genome_vs_baseline(best_genome, config)

    generation_num += 1

def eval_genome_vs_baseline(genome1, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
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
    with open(f"./data/trial/{generation_num}_genome.pkl", 'wb') as file:
        pickle.dump(genome1, file)
    if genome1_reward > baseline_reward:
        pgx.save_svg_animation(states, f"./data/trial/{generation_num}_trial_WIN.svg", frame_duration_seconds=0.2)
    else:
        pgx.save_svg_animation(states, f"./data/trial/{generation_num}_trial_LOSS.svg", frame_duration_seconds=0.2)

def eval_genome_vs_genome(genome1, genome2, config, save_match):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
    env, init, step = create_env()
    batch_size = 1
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    state = init(keys)
    genome1_reward = 0
    genome2_reward = 0
    if save_match:
        states = [state]

    while not (state.terminated | state.truncated).all():
        observations = np.array(state.observation)
        legal_actions = np.array(state.legal_action_mask)
        actions = []

        for i, (obs, legal) in enumerate(zip(observations, legal_actions)):
            if i % 2 == 0:
                output = net1.activate(obs.flatten().tolist())
            else:
                output = net2.activate(obs.flatten().tolist())
            action = np.argmax(output * legal)
            actions.append(action)

        actions = np.array(actions)
        state = step(state, actions)
        if save_match:
            states.append(state)
        rewards = state.rewards
        genome1_reward += jnp.sum(rewards[:, 0])
        genome2_reward += jnp.sum(rewards[:, 1])

    if save_match and batch_size == 1:
        global generation_num
        pgx.save_svg_animation(states, f"./data/game/{generation_num}_game.svg", frame_duration_seconds=0.2)
    if genome1_reward > genome2_reward:
        return 1, 0
    elif genome2_reward > genome1_reward:
        return 0, 1
    else:
        return 0.5, 0.5

# Define the NEAT configuration
def run_neat(config_file):
    # Literally everything abt the NEAT algorithm is default, except the input/outputs sizes to fit the go space
    config = neat.Config(neat.DefaultGenome, Reproduction.OverwriteReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    try:
        p = neat.Checkpointer.restore_checkpoint('data/restore_point.pkl')
        #best_genome = p.best_genome
        #eval_genome_vs_baseline(best_genome, config)
    except Exception as e:
        print(e)
        p = neat.Population(config)  # Makes a population

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50, filename_prefix="./data/checkpoints/"))  # Records a checkpoint every X generations
    winner = p.run(eval_genomes, 600)
    print('\nBest genome:\n{!s}'.format(winner))  # Prints the best genome at end of training... not super helpful

if __name__ == '__main__':
    run_neat('neat_config.txt')  # Options for NEAT are stored in a .txt file for some reason