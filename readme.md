# Obstacle Course with Genetic Algorithm and Reinforcement Learning

This project demonstrates a hybrid approach combining **Genetic Algorithm (GA)** and **Reinforcement Learning (RL)** to navigate an obstacle course. The RL agents learn by updating their actions based on Q-Tables, while the Genetic Algorithm helps optimize their strategies across generations.

![Obstacle Course Simulation](images/simulation.gif)

## Project Structure
- **`config.py`**: Configuration settings for the simulation, including parameters for the environment, agents, and obstacles.
- **`course.py`**: Contains the implementation of the obstacle course, including terrain, obstacles, and rules for the course.
- **`creature.py`**: Defines the creatures (agents) used in the simulation. Each agent has its own Q-Table for reinforcement learning.
- **`main.py`**: Main script to run the simulation, which orchestrates the training, evolution, and testing phases.
- **`obstacle_config.py`**: Defines specific obstacle configurations and how they are loaded into the environment.
- **`video_utils.py`**: Contains utilities for recording and saving the simulation as a video.

## Genetic Algorithm

The **Genetic Algorithm** (GA) is used to evolve the creatures over multiple generations, optimizing their ability to complete the obstacle course. The algorithm follows these steps:

1. **Initialization**: A population of creatures is generated with random attributes.
2. **Fitness Evaluation**: Each creature's performance is evaluated based on how well they navigate the obstacle course.
3. **Selection**: The top-performing creatures are selected as parents for the next generation.
4. **Crossover & Mutation**: New creatures are created by crossing over attributes of parents and applying mutations to introduce variability.
5. **Replacement**: The population is updated with the new generation of creatures.

## Reinforcement Learning

**Reinforcement Learning** is used simultaneously with GA to fine-tune each creature's actions during its lifetime. Each creature has its own Q-Table, which maps state-action pairs to rewards. The process follows these steps:

1. **State Representation**: The creature observes the environment and determines its current state.
2. **Action Selection**: Based on its Q-Table, the creature selects an action (explore vs exploit).
3. **Action Execution**: The creature executes the action, and the environment responds with the next state and a reward.
4. **Q-Table Update**: The creature updates its Q-Table using the reward received and the future expected reward (Q-Learning algorithm).
5. **Repeat**: The creature continues this loop until it reaches a terminal state, Finishes the course or fails (Hitting Obstacles or Have no new move available).

The RL algorithm allows each creature to improve its decision-making in real time, while the GA optimizes across generations.

## Configuration (`config.py` and `obstacle_config.py`)

- **Simulation Parameters**: Defines parameters such as the number of generations, population size, mutation rate, and crossover rate for GA.
- **Reinforcement Learning Parameters**: Includes the learning rate, discount factor, and exploration rate (epsilon) for RL.
- **Obstacle Course Setup**: Specifies the layout and complexity of the obstacle course.

## Sample Simulation

You can start the simulation using the following command:

```bash 
python3 main.py