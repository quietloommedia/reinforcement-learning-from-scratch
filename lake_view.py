"""Course artwork over the real FrozenLake map and observations.

This renderer never chooses actions, changes rewards, or advances the environment.
The learning algorithm does not depend on it. Assets were made with Codex ImageGen.
"""
from pathlib import Path
import math
import pygame

ASSETS = Path(__file__).resolve().parent / "assets" / "lake-fantasy-v1"
PINE = (16, 45, 40)
IVORY = (244, 240, 230)
BRASS = (230, 189, 112)


class LakeView:
    def __init__(self, description, size=960):
        pygame.font.init()
        self.tiles = [[cell.decode() if isinstance(cell, bytes) else str(cell)
                       for cell in row] for row in description]
        self.rows = len(self.tiles)
        self.cols = len(self.tiles[0])
        if any(len(row) != self.cols for row in self.tiles):
            raise ValueError("Map rows must have equal length")
        if any(tile not in "SFHG" for row in self.tiles for tile in row):
            raise ValueError("Unsupported FrozenLake tile")
        self.size = size
        self.margin = round(size * .025)
        self.gap = round(size * .0083)
        self.cell = (size - 2 * self.margin - self.gap * (self.cols - 1)) // self.cols
        self.height = 2 * self.margin + self.rows * self.cell + self.gap * (self.rows - 1)
        self.font = pygame.font.Font(None, round(self.cell * .16))
        self.small = pygame.font.Font(None, round(self.cell * .125))
        # Smooth scaling uses original high-resolution assets, never enlarged pixel art.
        self.images = {name: pygame.image.load(str(ASSETS / f"{name}-v1.png"))
                       for name in ("ice", "hole", "agent", "goal")}
        self.floor = pygame.transform.smoothscale(self.images["ice"], (self.cell, self.cell))
        self.hole = pygame.transform.smoothscale(self.images["hole"], (round(self.cell*.92),)*2)
        self.agent = pygame.transform.smoothscale(self.images["agent"], (round(self.cell*.79),)*2)
        self.goal = pygame.transform.smoothscale(self.images["goal"], (round(self.cell*.74),)*2)
        self.mini_goal = pygame.transform.smoothscale(self.images["goal"], (round(self.cell*.42),)*2)
        self.last_state = None

    def rect(self, state):
        if not 0 <= int(state) < self.rows * self.cols:
            raise ValueError("State is outside the map")
        row, col = divmod(int(state), self.cols)
        return pygame.Rect(self.margin + col*(self.cell+self.gap),
                           self.margin + row*(self.cell+self.gap), self.cell, self.cell)

    def label(self, surface, text, position, light=False, small=False):
        font = self.small if small else self.font
        ink, fill = (IVORY, PINE) if light else (PINE, IVORY)
        rendered = font.render(text, True, ink)
        box = rendered.get_rect(topleft=position).inflate(12, 8)
        pygame.draw.rect(surface, fill, box, border_radius=5)
        surface.blit(rendered, position)

    def draw(self, state, previous_state=None, progress=1.0):
        state = int(state)
        self.rect(state)  # Validate before drawing or reporting a state.
        surface = pygame.Surface((self.size, self.height))
        surface.fill(PINE)
        for row in range(self.rows):
            for col in range(self.cols):
                index = row*self.cols + col
                tile = self.tiles[row][col]
                box = self.rect(index)
                surface.blit(self.floor, box)
                if tile == "H":
                    surface.blit(self.hole, self.hole.get_rect(center=box.center))
                if tile == "G":
                    marker = self.mini_goal if state == index else self.goal
                    center = (box.right-self.cell*.23, box.centery) if state == index else box.center
                    surface.blit(marker, marker.get_rect(center=center))
                if tile in "SGH":
                    self.label(surface, {"S":"START", "G":"GOAL", "H":"HOLE"}[tile],
                               (box.x+14, box.bottom-round(self.cell*.15)), light=True, small=True)
                self.label(surface, str(index), (box.x+14, box.y+12))
        current = self.rect(state)
        pygame.draw.rect(surface, BRASS, current.inflate(-6, -6), width=4, border_radius=5)
        center = (current.centerx-self.cell*.14, current.centery) if self.tiles[state//self.cols][state%self.cols] == "G" else current.center
        if previous_state is not None and int(previous_state) != state and progress < 1:
            previous = self.rect(previous_state)
            p = min(1.0, max(0.0, progress))
            eased = p*p*(3-2*p)
            center = (previous.centerx + (center[0]-previous.centerx)*eased,
                      previous.centery + (center[1]-previous.centery)*eased - math.sin(p*math.pi)*self.cell*.045)
        surface.blit(self.agent, self.agent.get_rect(center=center))
        # Visual interpolation connects two observed states; it does not invent a state.
        self.last_state = state
        return surface


class LakeWindow:
    def __init__(self, description, slippery=False):
        pygame.init()
        self.view = LakeView(description, size=800)
        self.screen = pygame.display.set_mode((800, self.view.height+115))
        pygame.display.set_caption("FrozenLake | QuietLoom course renderer")
        self.font = pygame.font.Font(None, 27)
        self.clock = pygame.time.Clock()
        self.slippery = slippery
        self.open = True
        self.previous_state = None

    def show(self, state, message, seconds=.75):
        condition = "SLIPPERY" if self.slippery else "DRY"
        labels = [
            (f"{condition} / state {int(state)}    {message}", self.view.height+13, IVORY),
            ("Custom artwork / actual Gymnasium observations", self.view.height+49, BRASS),
            ("Esc or close window to stop", self.view.height+81, (189, 201, 194))]
        started = pygame.time.get_ticks()
        deadline = started + round(seconds*1000)
        while self.open and pygame.time.get_ticks() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.open = False
            self.screen.fill(PINE)
            progress = min(1.0, (pygame.time.get_ticks()-started)/220)
            self.screen.blit(self.view.draw(state, self.previous_state, progress), (0, 0))
            for text, y, color in labels:
                self.screen.blit(self.font.render(text, True, color), (24, y))
            pygame.display.flip()
            self.clock.tick(60)
        self.previous_state = int(state)
        return self.open

    def close(self):
        pygame.quit()
