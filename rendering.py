#%%
import pygame
import numpy as np
from time import sleep


class PygameRenderer:
    def __init__(self, width=400, height=400, delay=0.1):
        self.delay = delay
        pygame.display.init()
        pygame.display.set_caption("Coingame")
        self.screen = pygame.display.set_mode((width, height))

    def render(self, coingame):
        img = coingame.to_image()
        img = np.flipud(np.rot90(img))
        surface = pygame.surfarray.make_surface(img)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        sleep(self.delay)

    def render_image(self, img):
        img = np.flipud(np.rot90(img))
        surface = pygame.surfarray.make_surface(img)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        sleep(self.delay)
