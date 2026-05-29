import pygame
import numpy as np
from environments.base_environment import CellState


# Colour palette
COLOURS = {
    "background":   (18, 18, 18),
    "unknown":      (40, 40, 40),
    "free":         (220, 220, 220),
    "free_dim":     (120, 120, 120),
    "occupied":     (50, 80, 180),
    "occupied_dim": (30, 50, 100),
    "contested":    (200, 60, 60),
    "wall_truth":   (30, 30, 80),
    "free_truth":   (240, 240, 240),
    "entry":        (80, 200, 120),
    "sentinel":     (255, 200, 0),
    "grid_line":    (30, 30, 30),
    "panel_border": (60, 60, 60),
    "text":         (200, 200, 200),
    "text_bright":  (255, 255, 255),
    "rubble":       (139, 100, 60),
    "basalt":       (60, 60, 80),
    "thermal":      (255, 80, 0),
    "thermal_dim":  (100, 40, 0)
}

SCOUT_COLOURS = [
    (255, 100, 100),  # Scout 0 — red
    (100, 255, 100),  # Scout 1 — green
    (100, 180, 255),  # Scout 2 — blue
    (255, 200, 50),   # Scout 3 — yellow
    (200, 100, 255),  # Scout 4 — purple
]


class Renderer:
    """
    Pygame-based dual-panel renderer for SentinelSwarm.

    Left panel:  Ground truth environment
    Right panel: Sentinel world model (built incrementally)
    Bottom bar:  Mission metrics and Scout status
    """

    def __init__(self, environment, cell_px=12):
        self.env = environment
        self.cell_px = cell_px
        self.width = environment.width
        self.height = environment.height

        panel_w = self.width * cell_px
        panel_h = self.height * cell_px
        bar_h = 48
        gap = 8

        self.panel_w = panel_w
        self.panel_h = panel_h
        self.bar_h = bar_h
        self.gap = gap

        screen_w = panel_w * 2 + gap * 3
        screen_h = panel_h + bar_h + gap * 3

        pygame.init()
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("SentinelSwarm")

        self.font_sm = pygame.font.SysFont("consolas", 11)
        self.font_md = pygame.font.SysFont("consolas", 13)
        self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)

        # Panel offsets
        self.left_x = gap
        self.right_x = panel_w + gap * 2
        self.panels_y = gap

    def _cell_rect(self, panel_x, x, y):
        """Return the pygame Rect for a cell in a given panel."""
        px = panel_x + x * self.cell_px
        py = self.panels_y + y * self.cell_px
        return pygame.Rect(px, py, self.cell_px - 1, self.cell_px - 1)

    def _draw_ground_truth(self):
        """Render the ground truth environment in the left panel."""
        from environments.base_environment import TerrainType

        for x in range(self.width):
            for y in range(self.height):
                rect = self._cell_rect(self.left_x, x, y)

                if self.env.ground_truth[x, y]:
                    colour = COLOURS["wall_truth"]
                else:
                    terrain = self.env.terrain[x, y]
                    if terrain == TerrainType.RUBBLE:
                        colour = COLOURS["rubble"]
                    elif terrain == TerrainType.SMOOTH:
                        colour = COLOURS["basalt"]
                    elif terrain == TerrainType.THERMAL:
                        colour = COLOURS["thermal"]
                    else:
                        # Show thermal gradient on normal cells
                        heat = self.env.thermal_field[x, y]
                        if heat > 0.05:
                            t = min(heat * 2, 1.0)
                            colour = self._lerp_colour(
                                COLOURS["free_truth"],
                                COLOURS["thermal_dim"],
                                t
                            )
                        else:
                            colour = COLOURS["free_truth"]

                pygame.draw.rect(self.screen, colour, rect)

        # Entry points
        for ex, ey in self.env.entry_points:
            if 0 <= ey < self.height:
                rect = self._cell_rect(self.left_x, ex, ey)
                pygame.draw.rect(self.screen, COLOURS["entry"], rect)

        # Sentinel
        sx, sy = self.env.sentinel_position
        sent_px = self.left_x + sx * self.cell_px
        sent_py = self.panels_y + self.panel_h + 2
        pygame.draw.circle(
            self.screen, COLOURS["sentinel"],
            (sent_px + self.cell_px // 2, sent_py + 4), 5
        )

        label = self.font_md.render("GROUND TRUTH", True, COLOURS["text"])
        self.screen.blit(label, (self.left_x + 4, self.panels_y + 4))

        # Draw entry points
        for ex, ey in self.env.entry_points:
            if 0 <= ey < self.height:
                rect = self._cell_rect(self.left_x, ex, ey)
                pygame.draw.rect(self.screen, COLOURS["entry"], rect)

        # Draw Sentinel position indicator
        sx, sy = self.env.sentinel_position
        sent_px = self.left_x + sx * self.cell_px
        sent_py = self.panels_y + self.panel_h + 2
        pygame.draw.circle(
            self.screen, COLOURS["sentinel"],
            (sent_px + self.cell_px // 2, sent_py + 4), 5
        )

        # Panel label
        label = self.font_md.render("GROUND TRUTH", True, COLOURS["text"])
        self.screen.blit(label, (self.left_x + 4, self.panels_y + 4))

    def _draw_world_model(self, world_model):
        """Render the Sentinel's world model in the right panel."""
        for x in range(self.width):
            for y in range(self.height):
                rect = self._cell_rect(self.right_x, x, y)
                cell = world_model.get((x, y))

                if cell is None or cell["state"] == CellState.UNKNOWN:
                    colour = COLOURS["unknown"]
                elif cell["state"] == CellState.CONTESTED:
                    colour = COLOURS["contested"]
                elif cell["state"] == CellState.OCCUPIED:
                    # Brightness scales with confidence
                    t = cell["confidence"]
                    colour = self._lerp_colour(
                        COLOURS["occupied_dim"], COLOURS["occupied"], t
                    )
                else:  # FREE
                    t = cell["confidence"]
                    colour = self._lerp_colour(
                        COLOURS["free_dim"], COLOURS["free"], t
                    )

                pygame.draw.rect(self.screen, colour, rect)

        # Panel label
        label = self.font_md.render("SENTINEL MODEL", True, COLOURS["text"])
        self.screen.blit(label, (self.right_x + 4, self.panels_y + 4))

    def _draw_scouts(self, scout_states, panel_x):
        """Draw Scout agents on a panel."""
        for scout in scout_states:
            x, y = scout["position"]
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            colour = SCOUT_COLOURS[scout["id"] % len(SCOUT_COLOURS)]
            cx = panel_x + x * self.cell_px + self.cell_px // 2
            cy = self.panels_y + y * self.cell_px + self.cell_px // 2

            # Dot size proportional to energy
            energy = scout.get("energy", 1.0)
            radius = max(2, int(5 * energy))
            pygame.draw.circle(self.screen, colour, (cx, cy), radius)
            pygame.draw.circle(self.screen, COLOURS["text_bright"], (cx, cy), radius, 1)

    def _draw_info_bar(self, metrics, scout_states):
        """Render the bottom information bar."""
        bar_y = self.panels_y + self.panel_h + self.gap
        bar_rect = pygame.Rect(
            self.gap, bar_y,
            self.panel_w * 2 + self.gap, self.bar_h
        )
        pygame.draw.rect(self.screen, (25, 25, 25), bar_rect)
        pygame.draw.rect(self.screen, COLOURS["panel_border"], bar_rect, 1)

        # Timestep and metrics
        t = metrics.get("timestep", 0)
        fidelity = metrics.get("fidelity", 0.0)
        coverage = metrics.get("coverage", 0.0)

        info = f"T:{t:04d}   Fidelity:{fidelity:.1f}%   Coverage:{coverage:.1f}%"
        surf = self.font_lg.render(info, True, COLOURS["text_bright"])
        self.screen.blit(surf, (self.gap + 8, bar_y + 6))

        # Scout energy indicators
        scout_x = self.gap + 8
        scout_y = bar_y + 26
        for scout in scout_states:
            sid = scout["id"]
            energy = scout.get("energy", 1.0)
            colour = SCOUT_COLOURS[sid % len(SCOUT_COLOURS)]
            label = f"S{sid}:{energy*100:.0f}%"
            surf = self.font_sm.render(label, True, colour)
            self.screen.blit(surf, (scout_x, scout_y))
            scout_x += 72

    def _lerp_colour(self, c1, c2, t):
        """Linear interpolation between two colours."""
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def render(self, world_model=None, scout_states=None, metrics=None):
        """
        Main render call. Call once per simulation timestep.

        Args:
            world_model: dict mapping (x,y) -> cell dict, or None for empty
            scout_states: list of scout state dicts, or None
            metrics: dict with timestep, fidelity, coverage, or None
        """
        if world_model is None:
            world_model = {}
        if scout_states is None:
            scout_states = []
        if metrics is None:
            metrics = {"timestep": 0, "fidelity": 0.0, "coverage": 0.0}

        self.screen.fill(COLOURS["background"])

        self._draw_ground_truth()
        self._draw_world_model(world_model)
        self._draw_scouts(scout_states, self.left_x)
        self._draw_scouts(scout_states, self.right_x)
        self._draw_info_bar(metrics, scout_states)

        pygame.display.flip()

    def handle_events(self):
        """
        Process pygame events. Returns False if window closed.
        Call this every frame to keep the window responsive.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        """Shut down pygame."""
        pygame.quit()