from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from collections import defaultdict
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

class Grid:
    # a grid of lines.
    # each line is defined by a single 2d point.  The line goes through this point
    # and is perpendicular to the ray from the origin to that point
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
                    if first_intersection is None and self.intersection_filter(self.intersections[i, j]):
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
                    if x != i and self.intersection_filter(self.intersections[i][x])
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

    def intersection_filter(self, intersection):
        if np.any(np.isnan(intersection)):
            return False
        return np.linalg.norm(intersection) < 1e8

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
                if not self.intersection_filter(self.intersections[i,j]):
                    continue
                if debug:
                    ax.plot(*self.intersections[i,j,:], 'o')
                    ax.annotate(f'{i},{j}', self.intersections[i,j,:])
                else:
                    pass
                    #ax.plot(*intersections[i,j,:], '.')
    
    def plot_rhombuses(self, ax, debug=False):
        patches = [r.patch() for r in self.rhombuses.values() if r.position is not None]
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
 
    def segment_set(self):
        # return set of deduplicated segments
        return set(chain(chain(*(
            r.segments() for r in self.rhombuses.values()
                if r.position is not None
        ))))

    def average_center(self):
        return np.mean(
            np.stack(r.position for r in self.rhombuses.values() if r.position is not None),
            axis=0
        )

class RegularGrid(Grid):
    def __init__(self, num_directions, num_lines, offsets=None):
        self.basis = get_basis(num_directions)
        if offsets is None:
            offsets = np.random.random(num_directions)
        self.offsets = offsets
        self.grid_range = np.arange(num_lines) - num_lines // 2

        lines = np.zeros(
            (len(self.basis), len(self.grid_range), 2),
            dtype=float
        )

        for i, (vector, offset) in enumerate(zip(self.basis, offsets)):
            for j, k in enumerate(self.grid_range):
                lines[i, j] = vector * (k + offset)

        lines = lines.reshape(-1,2)
        super().__init__(lines)

    def coordinates(self, point):
        return self.basis @ point - self.offsets
        
    def intersection_filter(self, intersection):
        coordinates = self.coordinates(intersection)
        in_range = np.all(
            (coordinates > min(self.grid_range) - 1) & 
            (coordinates < max(self.grid_range) + 1)
        )
        return super().intersection_filter(intersection) & in_range

@dataclass
class Rhomb:
    id: str  # an id based on which intersection the rhombus came from
    sides: dict  # side of the rhombus
    directions: dict  # direction of the line which the rhombus came from
    up_neighbors: dict = field(default_factory=dict)
    down_neighbors: dict = field(default_factory=dict)
    position: Optional[np.ndarray] = None
   
    def position_neighbors(self, cascade=False):
        # position the rhombuses neighbors based on this rhombuses position
        # if cascade is True then keep positioning the neighbors of those neighbors etc.
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

    def corners(self):
        # return the corners of the rhombus
        try:
            i, j = self.sides.keys()
        except ValueError:
            raise Exception(
                'Expected two sides in order to draw rhombus, but got {} sides'.format(
                    len(self.sides)
                )
            )
        return np.stack([
            self.sides[i] + self.sides[j],
            self.sides[i] - self.sides[j],
            - self.sides[i] - self.sides[j],
            - self.sides[i] + self.sides[j],
        ]) / 2  + self.position

    def patch(self):
        # return a matplotlib pathc for the rhombus
        return Polygon(self.corners(), closed=True)

    def draw_path(self):
        return '<polygon points="{}" fill="none" stroke="black"/>'.format(
            ' '.join(
                '{},{}'.format(*corner) for corner in self.corners()
            )
        )

    def angle(self):
        # smallest angle between the two sides
        return np.arccos(np.round(abs(np.dot(*self.sides.values())), decimals=12))

    def segments(self):
        corners = np.round(self.corners(), decimals=12)
        for point1, point2 in zip(corners[:-1], corners[1:]):
            yield frozenset((tuple(point1), tuple(point2)))
        yield frozenset((tuple(corners[0]), tuple(corners[-1])))


class PointIndex:
    # segments indexed by their points

    def __init__(self, segments):
        self.index = defaultdict(list)
        for segment in segments:
            for point in segment:
                self.index[point].append(segment)
        self.index = dict(self.index)

    def __bool__(self):
        return bool(self.index)

    def __getitem__(self, arg):
        return self.index[arg]

    def get_nearest_point(self, point):
        return min(self.index.keys(), key=lambda x: distance(x, point))

    def pop(self, point):
        try:
            segments = self.index[point]
            segment = segments.pop()
            if len(segments) == 0:
                del self.index[point]
            other_point = [p for p in segment if p != point][0]

            segments = self.index[other_point]
            segments.remove(segment)
            if len(segments) == 0:
                del self.index[other_point]
            return segment
        except (IndexError, KeyError, ValueError) as e:
            return None

    def __len__(self):
        return sum(len(item) for item in self.index.values())

    def sorted_cuts(self, starting_point=(0,0)):
        index_copy = self.index.copy()
        current_point = starting_point
        cuts = []
        while self.index:
            current_point = self.get_nearest_point((current_point))
            current_cut = []
            seg = self.pop(current_point)
            while seg:
                current_cut.append(current_point)
                current_point, = (p for p in seg if p != current_point)
                seg = self.pop(current_point)

            current_cut.append(current_point)
            cuts.append(current_cut)
        self.index = index_copy
        return cuts

def distance(p1, p2):
    return np.sqrt(sum((x1-x2) ** 2 for x1, x2 in zip(p1, p2)))

def cuts_to_path(cuts, scale=1, pre_translate=(0, 0), post_translate=(0,0)):
    transform = lambda point: tuple([scale*(x + y) + z for x, y, z in zip(point, pre_translate, post_translate)])
    commands = []
    for cut in cuts:
        commands.append('M {} {}'.format(*transform(cut[0])))
        for point in cut[1:]:
            commands.append('L {} {}'.format(*transform(point)))
    return '\n'.join(commands)


if __name__ == '__main__':
    grid = RegularGrid(5, 5, offsets=[0.15, 0.15, 0.15, 0.15, 0.15])
    center = grid.average_center()
    segments = grid.segment_set()
    point_index = PointIndex(segments)
    cuts = point_index.sorted_cuts()
    print(len(cuts))
    print('done')
    cutting_distance = 0
    movement_distance = 0
    cutter_position = cuts[0][0]
    print('cutting', cutting_distance)
    print('movement', movement_distance)
    with open('test.svg', 'w') as f:
        print('<svg version="1.1" '
              'width="16in" height="12in" '
              'xmlns="http://www.w3.org/2000/svg"> ',
              file=f
        )
        print('<path d="{}" fill="none" stroke="black" stroke-width="0.3"/>'.format(
                cuts_to_path(cuts, scale=60, pre_translate=-center, post_translate=(8,6))
            ),
            file=f
        )
        print('</svg>', file=f)
    #for cut in cuts:
    #    movement_distance += distance(cutter_position, cut[0])
    #    cutting_distance += len(cut)
    #    cutter_position = cut[-1]
    #    plt.plot(*zip(*cut))
    #plt.show()
