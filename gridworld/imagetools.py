#%%
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Union
from gridworld import Gridworld
from gridworld.legend import AGENT

GRID_SIZE = 40
FONT = ImageFont.truetype("font/opensans_cond.ttf", 16)


def build_image(gridworld: Gridworld, draw_agents=True):
    image, _ = _draw_image(gridworld, draw_agents)
    return np.array(image)


def build_trajectory_image(gridworld_name, trajectory):
    start_state = trajectory[0][:-1]
    g = Gridworld.from_vector(gridworld_name, start_state)
    image, canvas = _draw_image(g, draw_agents=False)

    _draw_circle(canvas, g.agent1_position, "", (180, 180, 180))

    for step in trajectory:
        action = step[-1:][0]
        state = list(step[:-1])
        agent_position = (
            state.index(AGENT) % g.n_cols,
            state.index(AGENT) // g.n_cols,
        )
        _draw_action_arrow(canvas, agent_position, action)

    return np.array(image)


def _draw_circle(
    canvas, loc: tuple[int, int], txt: str, color: Union[tuple[int, int, int], str]
):
    x_loc = loc[0] * GRID_SIZE + 5
    y_loc = loc[1] * GRID_SIZE + 5

    canvas.ellipse(
        (
            x_loc,
            y_loc,
            x_loc + GRID_SIZE - 10,
            y_loc + GRID_SIZE - 10,
        ),
        fill=color,
    )

    w, _ = canvas.textsize(txt, FONT)
    l = (GRID_SIZE - 10) // 2 - w // 2

    canvas.text((x_loc + l + 1, y_loc + 3), txt, color="white", font=FONT)


def _draw_coin(canvas, loc, n):
    _draw_circle(canvas, loc, str(n), (100, 100, 100))


def _draw_agent(canvas, loc, name: str):
    _draw_circle(canvas, loc, name, "black")


def _draw_action_arrow(canvas, loc, action):
    x_loc = loc[0] * GRID_SIZE + GRID_SIZE // 2
    y_loc = loc[1] * GRID_SIZE + GRID_SIZE // 2

    # Up
    if action == 0:
        x_loc = x_loc
        y_loc = y_loc - 7
        canvas.polygon(
            [(x_loc, y_loc), (x_loc - 5, y_loc + 15), (x_loc + 5, y_loc + 15)],
            fill="black",
        )

    # Right
    if action == 1:
        x_loc = x_loc - 7
        y_loc = y_loc - 5
        canvas.polygon(
            [(x_loc, y_loc), (x_loc, y_loc + 10), (x_loc + 15, y_loc + 5)],
            fill="black",
        )

    # Down
    if action == 2:
        x_loc = x_loc - 5
        y_loc = y_loc - 7
        canvas.polygon(
            [(x_loc, y_loc), (x_loc + 10, y_loc), (x_loc + 5, y_loc + 15)],
            fill="black",
        )

    # Left
    if action == 3:
        x_loc = x_loc - 7
        y_loc = y_loc
        canvas.polygon(
            [(x_loc, y_loc), (x_loc + 15, y_loc - 5), (x_loc + 15, y_loc + 5)],
            fill="black",
        )


def _draw_wall(canvas, loc):
    x_loc = loc[0] * GRID_SIZE
    y_loc = loc[1] * GRID_SIZE
    canvas.rectangle([x_loc, y_loc, x_loc + GRID_SIZE, y_loc + GRID_SIZE], fill="black")


def _draw_grid(canvas, width, height):
    for x in range(0, height, GRID_SIZE):
        canvas.line([(x, 0), (x, height)], fill="black")

    for y in range(0, width, GRID_SIZE):
        canvas.line([(0, y), (width, y)], fill="black")


def _draw_image(gridworld, draw_agents=True):
    width = gridworld.n_cols * GRID_SIZE + 1
    height = gridworld.n_rows * GRID_SIZE + 1

    image = Image.new("RGB", (width, height), (220, 220, 220))
    canvas = ImageDraw.Draw(image)

    _draw_grid(canvas, width, height)

    for loc in gridworld.wall_positions:
        _draw_wall(canvas, loc)

    for i, loc in enumerate(gridworld.coin_positions):
        if loc:
            _draw_coin(canvas, loc, i + 1)

    if draw_agents:
        _draw_agent(canvas, gridworld.agent1_position, "A")
        if gridworld.agent2_position:
            _draw_agent(canvas, gridworld.agent2_position, "B")

    return image, canvas
