import pygame
import random
import numpy as np
from collections import deque
import time


class MazeSolver:
    @staticmethod
    def flood_fill(maze, start, goal):
        height, width = maze.shape
        distances = np.full((height, width), float('inf'))
        distances[start[0]][start[1]] = 0
        queue = deque([start])
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        came_from = {(start[0], start[1]): None}

        while queue:
            current = queue.popleft()
            if current[0] == goal[0] and current[1] == goal[1]:
                break

            for dx, dy in dirs:
                next_x, next_y = current[0] + dx, current[1] + dy
                if (0 <= next_x < height and 0 <= next_y < width and
                        maze[next_x][next_y] != 1 and
                        distances[next_x][next_y] == float('inf')):
                    distances[next_x][next_y] = distances[current[0]][current[1]] + 1
                    queue.append((next_x, next_y))
                    came_from[(next_x, next_y)] = current

        path = []
        current = (goal[0], goal[1])
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1] if path and path[-1] == (start[0], start[1]) else []

    @staticmethod
    def a_star(maze, start, goal):
        height, width = maze.shape
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_idx = 0
            for i in range(1, len(frontier)):
                if frontier[i][0] < frontier[current_idx][0]:
                    current_idx = i

            current = frontier.pop(current_idx)[1]

            if current == goal:
                break

            for dx, dy in dirs:
                next_pos = (current[0] + dx, current[1] + dy)

                if (0 <= next_pos[0] < height and 0 <= next_pos[1] < width and
                        maze[next_pos[0]][next_pos[1]] != 1):
                    new_cost = cost_so_far[current] + 1

                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + heuristic(goal, next_pos)
                        frontier.append((priority, next_pos))
                        came_from[next_pos] = current

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return path[::-1] if path and path[-1] == start else []


class Sensor:
    def __init__(self, range_cells=3):
        self.range = range_cells

    def detect_walls(self, maze, position, direction):
        x, y = position
        dx, dy = direction
        distance = 0

        for i in range(1, self.range + 1):
            new_x, new_y = x + dx * i, y + dy * i
            if not (0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1]):
                return distance
            if maze[new_x][new_y] == 1:
                return distance
            distance += 1
        return distance


class MazeEnvironment:
    def __init__(self, width=20, height=20, cell_size=35):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.maze = np.zeros((height, width), dtype=int)
        self.mouse_pos = [0, 0]
        self.goal = [height - 1, width - 1]
        self.explored_maze = np.full((height, width), -1)
        self.path = []
        self.visited_cells = set()
        self.direction = (0, 1)
        self.maze_solved = False
        self.sensor = Sensor()
        self.generate_realistic_maze()
        self.update_sensors()

    def create_simple_maze(self):
        self.maze.fill(1)
        for i in range(self.height):
            self.maze[i, 0] = 0
        for j in range(self.width):
            self.maze[self.height - 1, j] = 0

        for _ in range(self.width * self.height // 4):
            x = random.randrange(0, self.height)
            y = random.randrange(0, self.width)
            self.maze[x, y] = 0

        self.maze[0, 0] = 0
        self.maze[self.height - 1, self.width - 1] = 0

    def generate_realistic_maze(self):
        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            self.maze.fill(1)
            start_pos = (0, 0)
            stack = [start_pos]
            visited = {start_pos}
            self.maze[0, 0] = 0

            while stack:
                current = stack[-1]
                neighbors = []

                for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                    next_x = current[0] + dx
                    next_y = current[1] + dy
                    if (0 <= next_x < self.height and 0 <= next_y < self.width and
                            (next_x, next_y) not in visited):
                        neighbors.append((next_x, next_y))

                if neighbors:
                    next_cell = random.choice(neighbors)
                    wall_x = (current[0] + next_cell[0]) // 2
                    wall_y = (current[1] + next_cell[1]) // 2

                    self.maze[wall_x][wall_y] = 0
                    self.maze[next_cell[0]][next_cell[1]] = 0

                    if random.random() < 0.15:
                        diag_x = wall_x + random.choice([-1, 1])
                        diag_y = wall_y + random.choice([-1, 1])
                        if (0 <= diag_x < self.height and 0 <= diag_y < self.width):
                            self.maze[diag_x][diag_y] = 0

                    visited.add(next_cell)
                    stack.append(next_cell)
                else:
                    stack.pop()

            # Create loops and additional paths
            for _ in range(self.width * self.height // 3):
                x = random.randrange(1, self.height - 1)
                y = random.randrange(1, self.width - 1)

                if random.random() < 0.2:
                    chamber_size = random.randint(2, 3)
                    for dx in range(chamber_size):
                        for dy in range(chamber_size):
                            new_x = x + dx
                            new_y = y + dy
                            if (0 <= new_x < self.height - 1 and
                                    0 <= new_y < self.width - 1):
                                self.maze[new_x][new_y] = 0

                if random.random() < 0.3:
                    direction = random.choice([(0, 1), (1, 0), (1, 1), (-1, 1)])
                    length = random.randint(2, 4)
                    for i in range(length):
                        new_x = x + direction[0] * i
                        new_y = y + direction[1] * i
                        if (0 <= new_x < self.height - 1 and
                                0 <= new_y < self.width - 1):
                            self.maze[new_x][new_y] = 0

            # Ensure start and goal areas are clear
            self.maze[0:2, 0:2] = 0
            self.maze[self.height - 2:, self.width - 2:] = 0

            # Verify path exists
            test_path = MazeSolver.flood_fill(self.maze, (0, 0),
                                              (self.height - 1, self.width - 1))
            if test_path:
                return

            attempt += 1

        self.create_simple_maze()

    def reset_environment(self):
        self.mouse_pos = [0, 0]
        self.explored_maze = np.full((self.height, self.width), -1)
        self.path = []
        self.visited_cells = set()
        self.direction = (0, 1)
        self.maze_solved = False
        self.generate_realistic_maze()
        self.update_sensors()

    def update_sensors(self):
        x, y = self.mouse_pos
        self.explored_maze[x][y] = 0
        self.visited_cells.add((x, y))

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for direction in directions:
            distance = self.sensor.detect_walls(self.maze, self.mouse_pos, direction)
            dx, dy = direction
            for i in range(1, distance + 1):
                new_x, new_y = x + dx * i, y + dy * i
                self.explored_maze[new_x][new_y] = 0
            if distance < self.sensor.range:
                wall_x, wall_y = x + dx * (distance + 1), y + dy * (distance + 1)
                if 0 <= wall_x < self.height and 0 <= wall_y < self.width:
                    self.explored_maze[wall_x][wall_y] = 1


class MicromouseSimulation:
    def __init__(self):
        pygame.init()
        self.env = MazeEnvironment()
        self.screen_width = self.env.width * self.env.cell_size
        self.screen_height = self.env.height * self.env.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Enhanced Micromouse Simulation")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.DARK_GRAY = (64, 64, 64)

        self.move_delay = 50
        self.last_move_time = 0
        self.maze_completion_delay = 2000
        self.last_maze_completion_time = 0
        self.current_algorithm = "flood"

        self.update_path()

    def draw(self):
        self.screen.fill(self.WHITE)

        # Draw explored maze
        for i in range(self.env.height):
            for j in range(self.env.width):
                x = j * self.env.cell_size
                y = i * self.env.cell_size

                if self.env.explored_maze[i][j] == 1:
                    pygame.draw.rect(self.screen, self.BLACK,
                                     (x, y, self.env.cell_size, self.env.cell_size))
                elif self.env.explored_maze[i][j] == 0:
                    pygame.draw.rect(self.screen, self.WHITE,
                                     (x, y, self.env.cell_size, self.env.cell_size))
                else:
                    pygame.draw.rect(self.screen, self.GRAY,
                                     (x, y, self.env.cell_size, self.env.cell_size))

                pygame.draw.rect(self.screen, self.DARK_GRAY,
                                 (x, y, self.env.cell_size, self.env.cell_size), 1)

        # Draw path
        if self.env.path:
            for i in range(len(self.env.path) - 1):
                start = self.env.path[i]
                end = self.env.path[i + 1]
                start_pos = (start[1] * self.env.cell_size + self.env.cell_size // 2,
                             start[0] * self.env.cell_size + self.env.cell_size // 2)
                end_pos = (end[1] * self.env.cell_size + self.env.cell_size // 2,
                           end[0] * self.env.cell_size + self.env.cell_size // 2)
                pygame.draw.line(self.screen, self.BLUE, start_pos, end_pos, 2)

        # Draw visited cells
        for cell in self.env.visited_cells:
            x = cell[1] * self.env.cell_size
            y = cell[0] * self.env.cell_size
            pygame.draw.rect(self.screen, self.YELLOW,
                             (x, y, self.env.cell_size, self.env.cell_size), 2)

        # Draw mouse
        mouse_x = self.env.mouse_pos[1] * self.env.cell_size
        mouse_y = self.env.mouse_pos[0] * self.env.cell_size
        center_x = mouse_x + self.env.cell_size // 2
        center_y = mouse_y + self.env.cell_size // 2

        pygame.draw.circle(self.screen, self.RED,
                           (center_x, center_y),
                           self.env.cell_size // 3)

        end_x = center_x + self.env.direction[1] * self.env.cell_size // 3
        end_y = center_y + self.env.direction[0] * self.env.cell_size // 3
        pygame.draw.line(self.screen, self.BLACK, (center_x, center_y), (end_x, end_y), 3)

        # Draw goal
        goal_x = self.env.goal[1] * self.env.cell_size
        goal_y = self.env.goal[0] * self.env.cell_size
        pygame.draw.rect(self.screen, self.GREEN,
                         (goal_x, goal_y, self.env.cell_size, self.env.cell_size))

        # Draw target symbol
        target_center_x = goal_x + self.env.cell_size // 2
        target_center_y = goal_y + self.env.cell_size // 2
        target_radius = self.env.cell_size // 4
        pygame.draw.circle(self.screen, self.BLACK,
                           (target_center_x, target_center_y), target_radius, 2)
        pygame.draw.circle(self.screen, self.BLACK,
                         (target_center_x, target_center_y),
                         target_radius // 2, 1)
    def update_path(self):
        if self.current_algorithm == "flood":
            self.env.path = MazeSolver.flood_fill(self.env.explored_maze,
                                                tuple(self.env.mouse_pos),
                                                tuple(self.env.goal))
        else:
            self.env.path = MazeSolver.a_star(self.env.explored_maze,
                                            tuple(self.env.mouse_pos),
                                            tuple(self.env.goal))

    def move_mouse(self):
        if self.env.path and len(self.env.path) > 1:
            next_pos = self.env.path[1]
            dx = next_pos[0] - self.env.mouse_pos[0]
            dy = next_pos[1] - self.env.mouse_pos[1]
            if (dx, dy) != (0, 0):
                self.env.direction = (dx, dy)

            self.env.mouse_pos = list(next_pos)
            self.env.update_sensors()
            self.update_path()

            if self.env.mouse_pos == self.env.goal:
                self.env.maze_solved = True
                self.last_maze_completion_time = pygame.time.get_ticks()
            return True
        return False

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.env.reset_environment()
                        self.update_path()
                    elif event.key == pygame.K_a:
                        self.current_algorithm = "astar" if self.current_algorithm == "flood" else "flood"
                        self.update_path()
                    elif event.key == pygame.K_SPACE:
                        self.move_delay = 0 if self.move_delay > 0 else 50

            if current_time - self.last_move_time >= self.move_delay:
                if self.env.maze_solved:
                    if current_time - self.last_maze_completion_time >= self.maze_completion_delay:
                        self.env.reset_environment()
                        self.update_path()
                else:
                    if not self.move_mouse():
                        self.update_path()
                self.last_move_time = current_time

            self.draw()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    sim = MicromouseSimulation()
    sim.run()