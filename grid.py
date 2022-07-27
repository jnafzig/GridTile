from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from itertools import chain
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time
import logging 
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.t0 = time.time()
        logger.info(f'starting {self.text}')
        return self.t0

    def __exit__(self, type, value, traceback):
        logger.info(f'{self.text} took {time.time()-self.t0} seconds')

def rot(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=np.float64)
rot90 = rot(np.pi/2)

def intersection(vec1, vec2):
    normal1 = vec1 / jnp.linalg.norm(vec1)
    direction1 = jnp.matmul(rot90, normal1)
    normal2 = vec2 / jnp.linalg.norm(vec2)
    direction2 = jnp.matmul(rot90, normal2)
    x = jnp.linalg.solve(
        jnp.stack([direction1, -direction2]).T,
        vec2-vec1
    )
    return x[0] * direction1 + vec1

get_jax_intersections = jax.vmap(
    jax.vmap(
        intersection,
        (0, None),
        0
    ),
    (None, 0),
    1
)
get_intersections = lambda x, y: np.array(get_jax_intersections(x, y))

def get_basis(N):
    return np.stack(
        [rot(2*np.pi * i / N) @ np.array([1, 0]) for i in range(N)]
    )

def intersection_filter(intersection):
    if np.any(np.isnan(intersection)):
        return False
    return np.linalg.norm(intersection) < 1e8

class Grid:
    def __init__(self, lines):
        self.lines = lines
        self.n = len(lines)
        self.norms = lines / np.linalg.norm(lines, axis=1)[..., np.newaxis]
        self.directions = (rot90 @ self.norms[..., np.newaxis]).squeeze()


        with Timer('finding intersections'):
            self.intersections = get_intersections(lines, lines)

        with Timer('creating rhombuses'):
            self.rhombuses = {}
            first_intersection = None
            for i in range(self.n):
                for j in range(i+1, self.n):
                    self.rhombuses[frozenset((i,j))] = Rhomb(
                        id=f'{i},{j}',
                        sides={
                            j: self.norms[i],
                            i: self.norms[j]
                        },
                        directions={
                            j: self.directions[j],
                            i: self.directions[i]
                        }
                    )
                    if first_intersection is None and intersection_filter(self.intersections[i, j]):
                        first_intersection = (i, j)

        with Timer('sorting intersections'):
            self.orderings = np.argsort(
                (self.directions[:, np.newaxis, :] * self.intersections).sum(axis=2),
                axis=1
            )

        with Timer('connecting rhombuses'):
            for i in range(self.n):
                ordering = self.orderings[i].tolist()
                ordering = [x for x in ordering 
                    if x != i and intersection_filter(self.intersections[i][x])
                ]
                for j1, j2 in zip(ordering[:-1], ordering[1:]):
                    rhomb1 = self.rhombuses[frozenset((i, j1))]
                    rhomb2 = self.rhombuses[frozenset((i, j2))]
                    rhomb1.up_neighbors[i] = rhomb2
                    rhomb2.down_neighbors[i] = rhomb1

        with Timer('positioning rhombuses'):
            positioned_rhomb = frozenset(first_intersection)
            self.rhombuses[positioned_rhomb].position = np.zeros(2)
            self.rhombuses[positioned_rhomb].position_neighbors(cascade=True)

    @classmethod
    def regular_grid(cls, num_directions, num_lines, offsets=None):
        basis = get_basis(num_directions)
        if offsets is None:
            offsets = np.random.random(num_directions)
        grid_range = np.arange(num_lines) - num_lines // 2

        lines = np.zeros((len(basis), len(grid_range), 2), dtype=float)

        for i, (vector, offset) in enumerate(zip(basis, offsets)):
            for j, k in enumerate(grid_range):
                lines[i, j] = vector * (k + offset)

        lines = lines.reshape(-1,2)
        return cls(lines)

    def plot_lines(self, ax, debug=False):
        for line, norm, direction in zip(self.lines, self.norms, self.directions):
            ax.axline(
                line,
                line + rot(np.pi/2) @ line
            )
            if debug:
                ax.arrow(*line, *direction, width=.04)

        for i in range(self.n):
            for j in range(i+1, self.n):
                if not intersection_filter(self.intersections[i,j]):
                    continue
                if debug:
                    ax.plot(*self.intersections[i,j,:], 'o')
                    ax.annotate(f'{i},{j}', self.intersections[i,j,:])
                else:
                    pass
                    #ax.plot(*intersections[i,j,:], '.')
    
    def plot_rhombuses(self, ax, debug=False):
        patches = [r.draw() for r in self.rhombuses.values() if r.position is not None]
        p = PatchCollection(patches, alpha=0.4, edgecolor='k')
        if debug:
            for rhomb in rhombuses.values():
                if rhomb.position is None:
                    continue
                ax.annotate(f'{rhomb.id}', rhomb.position)
                for key in rhomb.sides:
                    vec = rhomb.sides[key] * np.sign(rhomb.sides[key] @ rhomb.directions[key])
                    ax.arrow(*(rhomb.position-vec/2), *vec, width=.02)
                    #ax.arrow(*(rhomb.position), *rhomb.directions[key], width=.02, color='k')
        ax.add_collection(p)
    

@dataclass
class Rhomb:
    id: str
    sides: dict
    directions: dict
    up_neighbors: dict = field(default_factory=dict)
    down_neighbors: dict = field(default_factory=dict)
    position: Optional[np.ndarray] = None
   
    def position_neighbors(self, cascade=False):
        stack = [self]
        while stack:
            self = stack.pop()
            newly_positioned = []
            for key, neighbor in self.up_neighbors.items():
                if neighbor.position is not None:
                    continue
                vec1 = self.sides[key] * np.sign(self.sides[key] @ self.directions[key])
                vec2 = neighbor.sides[key] * np.sign(neighbor.sides[key] @ neighbor.directions[key])
                neighbor.position = self.position + (vec1 + vec2) / 2
                newly_positioned.append(neighbor)
            for key, neighbor in self.down_neighbors.items():
                if neighbor.position is not None:
                    continue
                vec1 = self.sides[key] * np.sign(self.sides[key] @ self.directions[key])
                vec2 = neighbor.sides[key] * np.sign(neighbor.sides[key] @ neighbor.directions[key])
                neighbor.position = self.position - (vec1 + vec2) / 2
                newly_positioned.append(neighbor)
            if cascade:
                stack.extend(newly_positioned)

    def neighbors(self):
        return chain(
            self.up_neighbors.values(),
            self.down_neighbors.values()
        )

    def draw(self):
        try:
            i, j = self.sides.keys()
        except ValueError:
            raise Exception(
                'Expected two sides in order to draw rhombus, but got {} sides'.format(
                    len(self.sides)
                )
            )
        rhombus = np.stack([
            self.sides[i] + self.sides[j],
            self.sides[i] - self.sides[j],
            - self.sides[i] - self.sides[j],
            - self.sides[i] + self.sides[j],
        ]) / 2 + self.position
        return Polygon(rhombus, closed=True)


if __name__ == '__main__':
    grid = Grid.regular_grid(5, 11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace':0, 'hspace':0})

    with Timer('plotting'):
        grid.plot_lines(ax1)
        grid.plot_rhombuses(ax2)

    ax1.axis('square')
    ax1.axis('off')

    ax2.axis('square')
    ax2.axis('off')
    plt.show()
