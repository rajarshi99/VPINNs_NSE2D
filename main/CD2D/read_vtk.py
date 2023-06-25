import numpy as np

def read_vtk(vtk_file_name = 'Mean_NRealisations_500.00052.vtk'):
    with open(vtk_file_name, 'r') as fin:
        vtk = fin.read()

    lines = np.array(vtk.split('\n'))

    for l_num in range(len(lines)):
        line = lines[l_num]
        values = line.split(' ')
        if len(values) == 3 and values[0] == 'POINTS' and values[2] == 'float':
            n_xy = int(values[1])
            xy_vals_beg = l_num + 1
            xy_vals_end = l_num + n_xy
            found_key_line = True
            break

    for l_num in range(len(lines)):
        line = lines[l_num]
        values = line.split(' ')
        if line == 'SCALARS C float':
            c_vals_beg = l_num + 2
            c_vals_end = l_num + 1 + n_xy

    xy_coords = np.array([ line.split(' ') for line in lines[xy_vals_beg:xy_vals_end+1] ])[:,:2].astype(np.float64)
    c_vals = np.array([ line.split(' ') for line in lines[c_vals_beg:c_vals_end+1] ]).astype(np.float64)

    ind = np.lexsort((xy_coords[:,0],xy_coords[:,1]))
    xy_coords = xy_coords[ind]
    c_vals = c_vals[ind]

    return xy_coords, c_vals
