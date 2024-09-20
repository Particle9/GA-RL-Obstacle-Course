import pygame
from config import *
from course import ObstacleCourse
from video_utils import ensure_dir, create_video, record_frame, cleanup_frames

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hybrid GA-RL Obstacle Course")

def main():
    clock = pygame.time.Clock()
    course = ObstacleCourse()
    running = True
    save_file = "best_creature.pkl"
    frame_number = 0
    
    if RECORD_SIMULATION:
        temp_dir = "temp_frames"
        ensure_dir(temp_dir)

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genetic Algorithm + Reinforcement Learning Obstacle Course")

    # Try to load the best creature at the start
    course.load_best_creature(save_file)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    course.save_best_creature(save_file)
                elif event.key == pygame.K_l:
                    course.load_best_creature(save_file)

        course.update()
        course.draw(screen)
        pygame.display.flip()

        
        if RECORD_SIMULATION:
            # Record every frame
            record_frame(screen, frame_number, temp_dir)
            frame_number += 1

        if all(creature.is_dead or creature.reached_goal or creature.steps == MAX_STEPS for creature in course.population):
            course.evolve()
            course.reset_population()

        clock.tick(FPS)

    # Save the best creature before quitting
    course.save_best_creature(save_file)

    
    if RECORD_SIMULATION:
        # Create video from saved frames
        create_video(temp_dir, "simulation_video.mp4")

        # Clean up temporary frames
        cleanup_frames(temp_dir)

    pygame.quit()

if __name__ == "__main__":
    main()