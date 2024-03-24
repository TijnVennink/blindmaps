import pygame
import numpy as np

class Wall:
    def __init__(self, x, y, height, width, mode):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.rect = pygame.rect(self.x, self.y, self.width, self.height)
        self.mode = mode


screen = pygame.display.set_mode((1200, 400))
box1 = Wall(10, 10, 100, 100, 2)
box2 = Wall(500, 500, 100, 100, 1)
box3 = Wall(250, 250, 200, 100, 3)
walls = [box1, box2, box3]

for wall in walls:
    if haptic.collide(wall):
        mode = wall.mode
        if mode == 1:  # force pertubation in positive X
            F[0] += 10
        elif mode == 2:  # sinus force pertubation (getting shaken around)
            amplitude = 10
            frequency = 1
            phase = 0
            F[1] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif mode == 3:  # force pertubation in both direction
            F += np.array([5, 5])
        elif mode == 4:
            amplitude = 10
            frequency = 1
            phase = 0
            F[0] += amplitude * np.cos(2 * np.pi * frequency * t + phase)
            F[1] += amplitude * np.sin(2 * np.pi * frequency * t + phase)


for i in walls:
    pygame.draw.rect(screen, (255, 255, 255, 128), i.rect)
