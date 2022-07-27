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
plt.rcParams["figure.figsize"] = (15, 15)

   
def rot(theta):
    return jnp.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=jnp.float64)

rot90 = rot(jnp.pi/2)

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
get_intersections = jax.vmap(
    jax.vmap(
        intersection,
        (0, None),
        0
    ),
    (None, 0),
    1
)


            
def norm(vector):
    return np.sum(vector ** 2) ** (1/2)
            
def normalize(vector):
    return vector / norm(vector)


def get_basis(N):
    return np.stack(
        [rot(2*np.pi * i / N) @ np.array([1, 0]) for i in range(N)]
    )

def rhombus(norm1, norm2, p):
    rhombus = np.stack([norm1, norm2 + norm1, norm2, 0*norm1]) + p
    return Polygon(rhombus, closed=True)

N = 5 
basis = get_basis(N)
offsets = np.random.random(N)
grid_range = range(-10,10)

lines = np.zeros((len(basis), len(grid_range), 2), dtype=float)

for i, (vector, offset) in enumerate(zip(basis, offsets)):
    for j, k in enumerate(grid_range):
        lines[i, j] = vector * (k + offset)

lines = lines.reshape(-1,2)
positioned_rhomb = frozenset((0,30))
#lines = np.random.random((20,2)) * 2 - 1

print(repr(lines))

#lines = np.array([
#    [-0.80486042, -0.6139645 ],
#    [-0.38972012, -0.23409338],
#    [ 0.4669395,  -0.52421947],]
#)
#lines = np.array([
#    [-0.82241089, -0.24806271],
#    [-0.40205683, -0.79968747],
#    [ 0.64082981, -0.64682302]
#])

#positioned_rhomb = frozenset((0,1))

#lines = np.random.random((20, 2))*2-1
normed = lines / np.linalg.norm(lines, axis=1)[..., np.newaxis]

intersections = get_intersections(lines, lines)


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
            print('positioning', self.id)
            newly_positioned = []
            for key, neighbor in self.up_neighbors.items():
                if neighbor.position is not None:
                    continue
                vec1 = self.sides[key] * np.sign(self.sides[key] @ self.directions[key])
                vec2 = neighbor.sides[key] * np.sign(neighbor.sides[key] @ neighbor.directions[key])
                neighbor.position = self.position + (vec1 + vec2) / 2
                newly_positioned.append(neighbor)
                print(neighbor.id, 'up')
            for key, neighbor in self.down_neighbors.items():
                if neighbor.position is not None:
                    continue
                vec1 = self.sides[key] * np.sign(self.sides[key] @ self.directions[key])
                vec2 = neighbor.sides[key] * np.sign(neighbor.sides[key] @ neighbor.directions[key])
                neighbor.position = self.position - (vec1 + vec2) / 2
                newly_positioned.append(neighbor)
                print(neighbor.id, 'down')
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

positions = np.full(shape=intersections.shape, fill_value=np.NaN)
side1 = np.full(shape=intersections.shape, fill_value=np.NaN)
side2 = np.full(shape=intersections.shape, fill_value=np.NaN)
norms = lines / jnp.linalg.norm(lines, axis=1, keepdims=True)
directions = (rot90 @ norms[..., jnp.newaxis]).squeeze()
orderings = (directions[np.newaxis, ...] * intersections).sum(axis=2)

rhombuses = {}
m, n, _ = intersections.shape
count = 0
for i in range(m):
    norm_i = lines[i] / jnp.linalg.norm(lines[i])
    direction_i = jnp.matmul(rot90, norm_i)
    for j in range(i+1, n):
        norm_j = lines[j] / jnp.linalg.norm(lines[j])
        direction_j = jnp.matmul(rot90, norm_j)
        rhombuses[frozenset((i,j))] = Rhomb(
            id=f'{i},{j}',
            sides={
                j: norm_i,
                i: norm_j
            },
            directions={
                j: direction_j,
                i: direction_i
            }
        )
        count += 1

def intersection_filter(intersection):
    if np.any(np.isnan(intersection)):
        return False
    return np.linalg.norm(intersection) < 1e8

for i in range(m):
    print(i)
    norm_i = lines[i] / jnp.linalg.norm(lines[i])
    direction_i = jnp.matmul(rot90, norm_i)
    ordering = np.argsort(intersections[i] @ direction_i).tolist()
    ordering = [x for x in ordering if x != i and intersection_filter(intersections[i][x])]
    for j1, j2 in zip(ordering[:-1], ordering[1:]):
        rhomb1 = rhombuses[frozenset((i, j1))]
        rhomb2 = rhombuses[frozenset((i, j2))]
        rhomb1.up_neighbors[i] = rhomb2
        rhomb2.down_neighbors[i] = rhomb1

rhombuses[positioned_rhomb].position = np.zeros(2)
rhombuses[positioned_rhomb].position_neighbors(cascade=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), gridspec_kw={'wspace':0, 'hspace':0})

debug = False
for line in lines:
    ax1.axline(
        line,
        line + rot(np.pi/2) @ line
    )
    norm = line / jnp.linalg.norm(line)
    direction = jnp.matmul(rot90, norm)
    if debug:
        ax1.arrow(*line, *direction, width=.04)

m, n, _ = intersections.shape
for i in range(m):
    for j in range(i+1, n):
        if not intersection_filter(intersections[i,j]):
            continue
        if debug:
            ax1.plot(*intersections[i,j,:], 'o')
            ax1.annotate(f'{i},{j}', intersections[i,j,:])
        else:
            ax1.plot(*intersections[i,j,:], '.')

ax1.axis('square')
ax1.axis('off')
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])

patches = [r.draw() for r in rhombuses.values() if r.position is not None]
p = PatchCollection(patches, alpha=0.4, edgecolor='k')
if debug:
    for rhomb in rhombuses.values():
        if rhomb.position is None:
            continue
        ax2.annotate(f'{rhomb.id}', rhomb.position)
        for key in rhomb.sides:
            vec = rhomb.sides[key] * np.sign(rhomb.sides[key] @ rhomb.directions[key])
            ax2.arrow(*(rhomb.position-vec/2), *vec, width=.02)
            #ax2.arrow(*(rhomb.position), *rhomb.directions[key], width=.02, color='k')
ax2.add_collection(p)
ax2.axis('square')
ax2.axis('off')
plt.show()
#    
