import numpy as np
import random
from config import *
from obstacle_config import *
import math


# Class that represents each instance in the generation
class Creature:

    # Initialize Creature
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.genes = np.random.uniform(-1, 1, MAX_STEPS * 2)
        self.fitness = 0
        self.steps = 0
        self.path = set()
        self.is_dead = False
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.closest_distance_to_goal = float('inf')
        self.reached_goal = False
        self.previous_positions = []
        self.visited_positions = set()
        self.direct_path_to_goal = False
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.epsilon = START_EPSILON

    def get_state(self):
        return (int(self.x // 50), int(self.y // 50))
    
    # Get a move that didn't result to a path the creature already traversed in that generation
    def get_valid_random_action(self):
        valid_actions = self.actions.copy()
        random.shuffle(valid_actions)
        for action in valid_actions:
            new_x = int(self.x + action[0] * 5)
            new_y = int(self.y + action[1] * 5)
            if (new_x, new_y) not in self.visited_positions:
                return action
        
        # Return None if there're no valid moves
        return None 

    # Get the best possible move a creature can traverse
    def get_best_valid_action(self, state):
        actions = sorted(self.q_table[state].items(), key=lambda x: x[1], reverse=True)
        for action, _ in actions:
            new_x = int(self.x + action[0] * 5)
            new_y = int(self.y + action[1] * 5)
            if (new_x, new_y) not in self.visited_positions:
                return action
        
        # Return None if there're no valid moves
        return None 

    # Get the action that the creature will take at each steps
    def get_action(self, state):
        # if self.direct_path_to_goal:
        #     return self.move_towards_goal()

        # Reduce Epsilon as each steps taken
        self.epsilon = self.epsilon * DECAY

        # Limit Minimum Learning Rate. Get either a random move or best possible move (based on Reinforcement Learning)
        if random.random() < max(MIN_EPSILON, self.epsilon):
            return self.get_valid_random_action()
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        return self.get_best_valid_action(state)
        
    
    # If goal can be reached from a straight line just go straight (Currently unused)
    def move_towards_goal(self):
        dx = GOAL[0] - self.x
        dy = GOAL[1] - self.y
        distance = math.hypot(dx, dy)

        # Stop if already at the goal
        if distance == 0:
            return None  
        move_distance = min(5, distance)
        new_x = self.x + (dx / distance) * move_distance
        new_y = self.y + (dy / distance) * move_distance
        
        # Check if the new position has been visited
        if (int(new_x), int(new_y)) in self.visited_positions:
            return self.get_valid_random_action()
        
        return (new_x - self.x, new_y - self.y)
    
    # Orchestrator that checked the state of creature and calculate their actions accordingly
    def move(self):
        if not self.is_dead and not self.reached_goal and self.steps < MAX_STEPS:
            state = self.get_state()
            
            # if self.direct_path_to_goal:
            if False:
                action = self.move_towards_goal()
                if action is None:
                    self.is_dead = True
                    return
                dx, dy = action
            else:
                action = self.get_action(state)
                if action is None:
                    self.is_dead = True
                    return
                dx, dy = action[0] * 5, action[1] * 5
            
            new_x = self.x + dx
            new_y = self.y + dy
            
            if (new_x < CREATURE_SIZE or new_x > WIDTH - CREATURE_SIZE or
                new_y < CREATURE_SIZE or new_y > HEIGHT - CREATURE_SIZE):
                self.is_dead = True
            else:
                self.x, self.y = new_x, new_y
                self.path.add((int(self.x), int(self.y)))
                self.visited_positions.add((int(self.x), int(self.y)))
                self.steps += 1
            
            self.last_state = state
            self.last_action = action
            
            self.previous_positions.append((self.x, self.y))
            if len(self.previous_positions) > 10:
                self.previous_positions.pop(0)
    
    def continuous_to_discrete_action(self, action):
        if isinstance(action, tuple) and len(action) == 2:
            dx, dy = action
            if abs(dx) > abs(dy):
                return (1, 0) if dx > 0 else (-1, 0)
            else:
                return (0, 1) if dy > 0 else (0, -1)
        return action  

    # Update Q-Table for each creature after each moves (for Reinforcement Learning)
    def update_q_table(self, reward, new_state):
        if self.last_state is not None and self.last_action is not None:
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = {action: 0 for action in self.actions}
            
            # Convert continuous action to discrete for Q-table update
            discrete_action = self.continuous_to_discrete_action(self.last_action)
            
            old_q = self.q_table[self.last_state][discrete_action]
            
            if new_state not in self.q_table:
                self.q_table[new_state] = {action: 0 for action in self.actions}
            
            max_future_q = max(self.q_table[new_state].values())
            
            new_q = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
            self.q_table[self.last_state][discrete_action] = new_q
    
    # Give penalty if creature stays in one place (currently a bit redundant since creature that can't take moves will die)
    def calculate_stagnation_penalty(self):
        if len(self.previous_positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(self.previous_positions)):
            x1, y1 = self.previous_positions[i-1]
            x2, y2 = self.previous_positions[i]
            total_distance += math.hypot(x2 - x1, y2 - y1)
        
        avg_distance = total_distance / (len(self.previous_positions) - 1)
        
        # Penalize more if average movement is less than 1 pixel per step
        return max(0, 1 - avg_distance) * 0.5

    # Calculate the score/fitness of each creatures
    def calculate_fitness(self, goal, obstacle_count, visited_cells):
        if self.is_dead:
            self.fitness = 0
            return

        # Reward for Distance between Creature and Goal (With Euclidian Distance)
        current_distance = math.hypot(self.x - goal[0], self.y - goal[1])
        distance_score = 1 / (current_distance + 1)
        
        # Reward for The Closest the creature has been to the goal (in this generation)
        self.closest_distance_to_goal = min(self.closest_distance_to_goal, current_distance)
        progress_score = (math.hypot(self.initial_x - goal[0], self.initial_y - goal[1]) - self.closest_distance_to_goal) / 100

        # Reward for exploring new areas
        exploration_score = len(self.path) / (WIDTH * HEIGHT) * 10

        # Penalty for  the number of Obstacles between self and goal (if we pull a straight line)
        obstacle_penalty = obstacle_count * 0.1

        # Reward for path that the creature hasn't taken before
        novelty_score = len(visited_cells) / (WIDTH * HEIGHT / 100) * 5

        # Penalty for staying in one place
        stagnation_penalty = self.calculate_stagnation_penalty()

        # Reward for surviving
        survival_bonus = self.steps / MAX_STEPS

        # Combine all components
        self.fitness = (
            distance_score +
            progress_score +
            exploration_score +
            novelty_score +
            survival_bonus -
            obstacle_penalty -
            stagnation_penalty
        )

        # Massive Reward for reaching the goal
        if self.reached_goal:
            self.fitness += GOAL_REWARD