#%%
import os

from gridworld import Gridworld
from gridworld.imagetools import build_image

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
import numpy as np
from time import sleep


class PygameRenderer:
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.screen = None

    def render(self, gridworld: Gridworld):
        img = build_image(gridworld)
        self.render_image(img)

    def render_image(self, img):
        if not self.screen:
            pygame.display.init()
            pygame.display.set_caption("Coingame")
            screen_size = img.shape
            self.screen = pygame.display.set_mode((screen_size[0], screen_size[1]))

        img = np.flipud(np.rot90(img))
        surface = pygame.surfarray.make_surface(img)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        sleep(self.delay)
