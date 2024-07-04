import os
os.environ['JAX_PLATFORMS'] = 'cpu' # This is bc I'm using my GPUs for xpilot, REMOVE THIS
import jax
import jax.numpy as jnp
import pgx
import neat
import numpy as np
import random

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
    if generation_num % 50 == 0 and generation_num > 0:
        batch_size = 10
        save = False
    elif generation_num-1 % 50 == 0:
        batch_size = 1
        save = True
    else:
        save = False
        batch_size = 1
    generation_num += 1
    random.shuffle(genomes)  # Shuffle genomes to ensure random pairing

    for i in range(0, len(genomes), 2):
        genome_id_1, genome_1 = genomes[i]
        genome_id_2, genome_2 = genomes[i + 1]
        genome_1.fitness, genome_2.fitness = eval_genome_vs_genome(genome_1, genome_2, config, batch_size, save)
        if save:
            save = False

def eval_genome_vs_genome(genome1, genome2, config, batch_size, save_match):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
    env, init, step = create_env()
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

    if save_match:
        global generation_num
        pgx.save_svg_animation(states, f"{generation_num}_game.svg", frame_duration_seconds=0.2)

    if genome1_reward > genome2_reward:
        return 1, 0
    elif genome2_reward > genome1_reward:
        return 0, 1
    else:
        return 0.5, 0.5

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
    winner = p.run(eval_genomes, 500)
    print('\nBest genome:\n{!s}'.format(winner))  # Prints the best genome at end of training... not super helpful

if __name__ == '__main__':
    run_neat('neat_config.txt')  # Options for NEAT are stored in a .txt file for some reason