#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np

GRID_SIZE = 40
FONT = ImageFont.truetype("font/opensans_cond.ttf", 16)


class ImageBuilder:
    def __init__(self, n_rows, n_cols):
        self.width = n_cols * GRID_SIZE + 1
        self.height = n_rows * GRID_SIZE + 1

        self.image = Image.new("RGB", (self.width, self.height), (220, 220, 220))
        self.canvas = ImageDraw.Draw(self.image)

    def draw_circle(self, loc, txt, color):
        x_loc = loc[0] * GRID_SIZE + 5
        y_loc = loc[1] * GRID_SIZE + 5

        self.canvas.ellipse(
            (
                x_loc,
                y_loc,
                x_loc + GRID_SIZE - 10,
                y_loc + GRID_SIZE - 10,
            ),
            fill=color,
        )

        w, _ = self.canvas.textsize(txt, FONT)
        l = (GRID_SIZE - 10) // 2 - w // 2

        self.canvas.text((x_loc + l + 1, y_loc + 3), txt, color="white", font=FONT)

    def draw_coin(self, loc, n):
        self.draw_circle(loc, str(n), (100, 100, 100))

    def draw_agent(self, loc):
        self.draw_circle(loc, "A", "black")

    def draw_action_arrow(self, loc, action):
        x_loc = loc[0] * GRID_SIZE + GRID_SIZE // 2
        y_loc = loc[1] * GRID_SIZE + GRID_SIZE // 2

        # Up
        if action == 0:
            x_loc = x_loc
            y_loc = y_loc - 7
            self.canvas.polygon(
                [(x_loc, y_loc), (x_loc - 5, y_loc + 15), (x_loc + 5, y_loc + 15)],
                fill="black",
            )

        # Right
        if action == 1:
            x_loc = x_loc - 7
            y_loc = y_loc - 5
            self.canvas.polygon(
                [(x_loc, y_loc), (x_loc, y_loc + 10), (x_loc + 15, y_loc + 5)],
                fill="black",
            )

        # Down
        if action == 2:
            x_loc = x_loc - 5
            y_loc = y_loc - 7
            self.canvas.polygon(
                [(x_loc, y_loc), (x_loc + 10, y_loc), (x_loc + 5, y_loc + 15)],
                fill="black",
            )

        # Left
        if action == 3:
            x_loc = x_loc - 7
            y_loc = y_loc
            self.canvas.polygon(
                [(x_loc, y_loc), (x_loc + 15, y_loc - 5), (x_loc + 15, y_loc + 5)],
                fill="black",
            )

    def draw_grid(self):
        for x in range(0, self.height, GRID_SIZE):
            self.canvas.line([(x, 0), (x, self.height)], fill="black")

        for y in range(0, self.width, GRID_SIZE):
            self.canvas.line([(0, y), (self.width, y)], fill="black")

    def get_image(self, coingame):

        self.draw_grid()

        for i, loc in enumerate(coingame.coin_positions):
            if loc:
                self.draw_coin(loc, i + 1)

        self.draw_agent(coingame.agent_position)

        return np.array(self.image)
