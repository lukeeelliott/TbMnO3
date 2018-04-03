import argparse
import itertools
import numpy as np
from mantid.geometry import SpaceGroupFactory, CrystalStructure

# Parse command line
parser = argparse.ArgumentParser(description='Output VAMPIRE unit cell to file')
parser.add_argument('filename', metavar='f', type=str)
args = parser.parse_args()

# String joins
s = ' '
t = ';'

# Space group taken from literature
sg = SpaceGroupFactory.createSpaceGroup("P b n m")

# Unit cell dimensions a, b, c
unit_dim = [5.3170, 5.8348, 7.4202]

# Atom positions
Tb_position = [0.9973, 0.0772, 0.25]
Mn_position = [0.5, 0.0, 0.0]
O1_position = [0.1108, 0.4685, 0.25]
O2_position = [0.7086, 0.3272, 0.0504]

# Generate fractional positions
Tb_equiv = sg.getEquivalentPositions(Tb_position)
Mn_equiv = sg.getEquivalentPositions(Mn_position)

# Generate all O in 3x3
O1_equiv = sg.getEquivalentPositions(O1_position)
O2_equiv = sg.getEquivalentPositions(O2_position)

# Oxygen container
O_all = []

for pos in O1_equiv:
    for i, j, k in itertools.product(xrange(-1, 1, 1), xrange(-1, 1, 1), xrange(-1, 1, 1)):
        O_all.append(np.array([sum(x) for x in zip(pos, [i,j,k])]))

for pos in O2_equiv:
    for i, j, k in itertools.product(xrange(-1, 1, 1), xrange(-1, 1, 1), xrange(-1, 1, 1)):
        O_all.append(np.array([sum(x) for x in zip(pos, [i,j,k])]))


# Exchange vectors
Jaa = np.array([[1., 0., 0.]])
Jbb = np.array([[0., 1., 0.]])
Jcc = np.array([[0., 0., 0.5]])
Jab = np.array([[0.5, 0.5, 0.]])

# Rotation matrices
A_180 = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
B_180 = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
C_90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

# Calculate all exchanges
symmetric_exchange = np.array([
    np.append(Jaa, 0.0913e-21), np.append(np.dot(Jaa, B_180), 0.0913e-21),
    np.append(Jbb, 0.1362e-21), np.append(np.dot(Jbb, A_180), 0.1362e-21),
    np.append(Jcc, 0.0801e-21), np.append(np.dot(Jcc, A_180), 0.0801e-21),
    np.append(Jab, -0.2435e-21), np.append(np.dot(Jab, C_90), -0.2435e-21),
    np.append(np.dot(Jab, -C_90), -0.2435e-21), np.append(np.dot(Jab, np.dot(C_90, C_90)), -0.2435e-21)
])

# Scaling factor for DMI
scale = 0.2

# Multi-exchange container
all_exchange = []
count = 0

# Step through Mn atoms
for i, pos in enumerate(Mn_equiv):
    # Single exchange container
    this_exchange = []

    # Step through all exchanges
    for ind, exchange in enumerate(symmetric_exchange):

        # Apply symmetric exchange and map onto unit cell
        exchange_atom_pos = [np.mod(pos+exchange[:3], 1.0), (np.floor(pos+exchange[:3])).astype(int)]

        # Find closest Oxygen atom
        mid_point = pos + 0.5 * exchange[:3]
        dis = 1000
        for pos_3 in O_all:
            if np.linalg.norm(mid_point-pos_3) < dis:
                dis = np.linalg.norm(mid_point-pos_3)
                O_vec = pos_3-mid_point
                O_pos = pos_3

        # Calculate DMI vector
        D_vec = np.cross(exchange[:3], O_vec)

        # Form exchange tensor
        # Some quirks with zeros, extremely ugly code incoming
        exchange_tensor = np.zeros((3,3))
        exchange_tensor = [0.0]*9
        exchange_tensor[1] = scale*exchange[-1]*D_vec[2]
        exchange_tensor[3] = scale*exchange[-1]*D_vec[2]
        exchange_tensor[2] = scale*exchange[-1]*D_vec[1]
        exchange_tensor[6] = scale*exchange[-1]*D_vec[1]
        exchange_tensor[5] = -1.*scale*exchange[-1]*D_vec[0]
        exchange_tensor[7] = scale*exchange[-1]*D_vec[0]

        if 0 <= ind <= 1:
            exchange_tensor[0] = exchange[-1]
        elif 2 <= ind <= 3:
            exchange_tensor[4] = exchange[-1]
        elif 4 <= ind <= 5:
            exchange_tensor[8] = exchange[-1]
        elif 6 <= ind <= 9:
            exchange_tensor[1] = exchange[-1]
            exchange_tensor[3] = exchange[-1]

        for j, pos_2 in enumerate(Mn_equiv):
            diff = np.linalg.norm(pos_2 - exchange_atom_pos[0])
            if diff == 0.:
                this_exchange.append([count, int(i), int(j), exchange_atom_pos[1][0],
                                      exchange_atom_pos[1][1], exchange_atom_pos[1][2], s.join(map(str, exchange_tensor)).strip('[').strip(']')])
                count += 1

    all_exchange.append(this_exchange)

# File writing
f = open(args.filename, 'w')

# Hardcoded headers
f.write('# Unit cell size:\n')
f.write(s.join([s.join(map(str, unit_dim)), '\n']))
f.write('# Unit cell vectors:\n')
f.write('1.0 0.0 0.0\n')
f.write('0.0 1.0 0.0\n')
f.write('0.0 0.0 1.0\n')
f.write('# Atoms: num_atoms num_materials; id cx cy cz mat cat hcat\n')
f.write(str(len(Mn_equiv)+len(O1_equiv)+len(O2_equiv)) + ' 1\n')

# Atom positions
for i, pos in enumerate(Mn_equiv):
    f.write(s.join([str(i), s.join(map(str, pos)), '0', '0', '0\n']))

# Exchange interactions
f.write('# Interactions n exctype; id i j dx dy dz Jij\n')
f.write(s.join([str(count), '2', '\n']))
for i in all_exchange:
    for j in i:
        f.write(s.join([s.join(map(str, j)), '\n']))
