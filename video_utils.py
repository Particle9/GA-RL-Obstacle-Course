import os
import cv2
from config import *
import pygame


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def record_frame(screen, frame_number, directory):
    # Create a new surface with the desired recording size
    scaled_surface = pygame.Surface((RECORD_WIDTH, RECORD_HEIGHT))
    # Scale down the screen surface to the recording size
    pygame.transform.scale(screen, (RECORD_WIDTH, RECORD_HEIGHT), scaled_surface)
    # Save the scaled surface
    pygame.image.save(scaled_surface, f"{directory}/frame_{frame_number:08d}.png")

def create_video(input_directory, output_filename, fps=60):
    images = [img for img in os.listdir(input_directory) if img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(input_directory, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_directory, image)))

    video.release()
    print(f"Video saved as {output_filename}")

def cleanup_frames(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
    os.rmdir(directory)
    print(f"Cleaned up temporary frames in {directory}")