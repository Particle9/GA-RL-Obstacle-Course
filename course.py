from creature import Creature
from config import *
from obstacle_config import *
import pygame
import math
import random
import pickle
import os
import numpy as np

# Class that represent the Obstacle Course the creature must traverse
class ObstacleCourse:

    # Initialize Obstacles and the first generation
    def __init__(self):
        self.population = [Creature(*START) for _ in range(POPULATION_SIZE)]
        self.obstacles = [pygame.Rect(*obs) for obs in OBSTACLES]
        self.goal = GOAL
        self.generation = 0
        self.best_creature = None
        self.best_fitness = float('-inf')
        self.visited_cells = set()
        self.creatures_reached_goal = 0
        self.goal_reaching_creatures = []

    # Update each state
    def update(self):
        current_best_fitness = float('-inf')
        current_best_creature = None

        for creature in self.population:
            if not creature.is_dead and not creature.reached_goal:
                # Check if there's a direct path to the goal
                creature.direct_path_to_goal = self.check_direct_path(creature, self.goal)
                
                old_state = creature.get_state()
                old_x, old_y = creature.x, creature.y
                creature.move()
                new_state = creature.get_state()
                
                self.visited_cells.add(new_state)
                
                # Small negative reward for each step
                reward = -1  
                if creature.is_dead:
                    # Large negative reward for dying
                    reward = -100  
                elif self.check_collision(creature, old_x, old_y):
                    # Large negative reward for hitting an obstacle
                    creature.is_dead = True
                    creature.x, creature.y = old_x, old_y
                    reward = -100  
                elif math.hypot(creature.x - self.goal[0], creature.y - self.goal[1]) < CREATURE_SIZE:
                    # Large positive reward for reaching the goal
                    creature.reached_goal = True
                    self.creatures_reached_goal += 1
                    reward = GOAL_REWARD  
                
                # Penalty for staying in one place
                stagnation_penalty = creature.calculate_stagnation_penalty() * 10
                reward -= stagnation_penalty
                
                creature.update_q_table(reward, new_state)
                
                # Count obstacles between creature and goal (in a straight line)
                obstacle_count = self.count_obstacles_between(creature, self.goal)
                creature.calculate_fitness(self.goal, obstacle_count, self.visited_cells)

                if creature.fitness > current_best_fitness:
                    current_best_fitness = creature.fitness
                    current_best_creature = creature

        # Update the best creature
        if current_best_creature and current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_creature = current_best_creature
    
    # Check if there's a direct path between creature and goal
    def check_direct_path(self, creature, goal):
        start = (creature.x, creature.y)
        for obstacle in self.obstacles:
            if obstacle.clipline(start, goal):
                return False
        return True
    
    # Check for collision between Creature and Obstacle
    def check_collision(self, creature, old_x, old_y):
        creature_rect = pygame.Rect(creature.x - CREATURE_SIZE, creature.y - CREATURE_SIZE, 
                                    CREATURE_SIZE * 2, CREATURE_SIZE * 2)
        for obstacle in self.obstacles:
            if creature_rect.colliderect(obstacle):
                return True
        
        for obstacle in self.obstacles:
            if obstacle.clipline((old_x, old_y), (creature.x, creature.y)):
                return True
        
        return False
    
    # Count Obstacles between creature and the goal
    def count_obstacles_between(self, creature, goal):
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                return False
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                return True
            return False

        count = 0
        creature_pos = (creature.x, creature.y)
        for obstacle in self.obstacles:
            obstacle_lines = [
                (obstacle.left, obstacle.top, obstacle.right, obstacle.top),
                (obstacle.right, obstacle.top, obstacle.right, obstacle.bottom),
                (obstacle.right, obstacle.bottom, obstacle.left, obstacle.bottom),
                (obstacle.left, obstacle.bottom, obstacle.left, obstacle.top)
            ]
            
            for line in obstacle_lines:
                if line_intersection((*creature_pos, *goal), line):
                    count += 1
                    break  # Count each obstacle only once
        
        return count
    
    # Draw the scene on Pygame
    def draw(self, screen):
        screen.fill(WHITE)
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, BLACK, obstacle)
        pygame.draw.circle(screen, GREEN, self.goal, CREATURE_SIZE)
        
        for creature in self.population:
            if creature == self.best_creature:
                color = BLUE  # Best creature in purple
            elif creature.reached_goal:
                color = YELLOW
            elif creature.is_dead:
                color = GRAY
            else:
                color = RED
            pygame.draw.circle(screen, color, (int(creature.x), int(creature.y)), CREATURE_SIZE)
            for point in creature.path:
                screen.set_at(point, (255, 200, 200))
        
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {self.generation}", True, BLUE)
        screen.blit(gen_text, (10, 10))
        best_text = font.render(f"Best Fitness: {self.best_fitness:.4f}", True, BLUE)
        screen.blit(best_text, (10, 50))
        alive_text = font.render(f"Alive: {sum(1 for c in self.population if not c.is_dead)}/{POPULATION_SIZE}", True, BLUE)
        screen.blit(alive_text, (10, 90))
        goal_text = font.render(f"Reached Goal: {self.creatures_reached_goal}/{POPULATION_SIZE}", True, BLUE)
        screen.blit(goal_text, (10, 130))

    # Create New Generation based on the best performing creatures (Also keep top 2 of the best performing creature)
    def evolve(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elite = self.population[:2]
        new_population = []
        
        while len(new_population) < POPULATION_SIZE - 2:
            parent1, parent2 = random.choices(self.population[:20], k=2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = elite + new_population
        self.generation += 1
        # Reset for the new generation
        self.creatures_reached_goal = 0  

    # Make a child for the parents chosen in the evolve function
    def crossover(self, parent1, parent2):
        child = Creature(WIDTH // 2, HEIGHT - 50)
        mask = np.random.rand(MAX_STEPS * 2) < 0.5
        child.genes = np.where(mask, parent1.genes, parent2.genes)
        # Inherit Q-tables from parents
        child.q_table = {**parent1.q_table, **parent2.q_table}
        return child
    
    # Random Mutation for each children
    def mutate(self, creature):
        mask = np.random.rand(MAX_STEPS * 2) < MUTATION_RATE
        creature.genes[mask] = np.random.uniform(-1, 1, mask.sum())
        # Mutate Q-table
        for state in creature.q_table:
            for action in creature.q_table[state]:
                if random.random() < MUTATION_RATE:
                    creature.q_table[state][action] += np.random.normal(0, 0.1)

    # Reset every creature in the population scores
    def reset_population(self):
        self.visited_cells.clear()
        for creature in self.population:
            creature.x = WIDTH // 2
            creature.y = HEIGHT - 50
            creature.initial_x = creature.x
            creature.initial_y = creature.y
            creature.steps = 0
            creature.path.clear()
            creature.is_dead = False
            creature.reached_goal = False
            creature.last_state = None
            creature.last_action = None
            creature.closest_distance_to_goal = float('inf')
            creature.visited_positions.clear()
            creature.epsilon = START_EPSILON

    # Save the best creature in pickle file
    def save_best_creature(self, filename):
        if self.best_creature:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_creature, f)
            print(f"Best creature saved to {filename}")

    # Load best creature if exist
    def load_best_creature(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_creature = pickle.load(f)
            self.population[0] = loaded_creature
            self.best_creature = loaded_creature
            self.best_fitness = loaded_creature.fitness
            print(f"Best creature loaded from {filename}")
        else:
            print(f"File {filename} not found")