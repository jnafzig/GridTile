from grid import *

grid = RegularGrid(5, 5, offsets=[0.15, 0.15, 0.15, 0.15, 0.15])
center = grid.average_center()
relative_positions = grid.positions() - center
distances = np.linalg.norm(relative_positions, axis=1)
max_distance = np.max(distances)

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
with open('puzzle01.svg', 'w') as f:
    print('<svg version="1.1" '
          'width="16in" height="12in" '
          'xmlns="http://www.w3.org/2000/svg"> ',
          file=f
    )
    print('<path d="{}" fill="none" stroke="red" stroke-width="0.3"/>'.format(
            cuts_to_path(cuts, scale=60, pre_translate=-center, post_translate=(96*6,96*6))
        ),
        file=f
    )
    print('<circle cx="{}" cy="{}" r="{}" fill="none" stroke="red" stroke-width="0.3"/>'.format(
            96*6, 96*6, 1.1*max_distance*60
        ),
        file=f
    )
    print('</svg>', file=f)
