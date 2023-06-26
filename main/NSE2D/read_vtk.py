import numpy as np
import matplotlib.pyplot as plt

def read_vtk(vtk_file_name = 'Mean_NRealisations_500.00052.vtk'):
    with open(vtk_file_name, 'r') as fin:
        vtk = fin.read()

    lines = np.array(vtk.split('\n'))

    for l_num in range(len(lines)):
        line = lines[l_num]
        line = line.strip() 
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
        if line == 'SCALARS p float':
            p_vals_beg = l_num + 2
            p_vals_end = l_num + 1 + n_xy
        if line == 'VECTORS u float':
            uv_vals_beg = l_num + 1
            uv_vals_end = l_num + n_xy

    xy_coords = np.array([ line.split(' ') for line in lines[xy_vals_beg:xy_vals_end+1] ])[:,:2].astype(np.float64)
    uv_vals = np.array([ line.split(' ') for line in lines[uv_vals_beg:uv_vals_end+1] ])[:,:2].astype(np.float64)
    p_vals = np.array([ line.split(' ') for line in lines[p_vals_beg:p_vals_end+1] ]).astype(np.float64)

    ind = np.lexsort((xy_coords[:,0],xy_coords[:,1]))
    xy_coords = xy_coords[ind]
    uv_vals = uv_vals[ind]
    p_vals = p_vals[ind]

    return xy_coords, uv_vals[:,0], uv_vals[:,1], p_vals

# Function genertates the co-ordinates for ghia's benchmark on x direction
def generate_ghia_point_u1():
    data = np.genfromtxt("../../ghia_u.csv")
    
    y = data[:,0]
    x = np.ones_like(y)*0.5
    
    # concatenate both columns side by side
    coords = np.vstack((x,y)).T
    
    return coords

# Function genertates the co-ordinates for ghia's benchmark on y direction
def generate_ghia_point_u2():
    data = np.genfromtxt("../../ghia_v.csv")
    
    x = data[:,0]
    y = np.ones_like(x)*0.5
    
    # concatenate both columns side by side
    coords = np.vstack((x,y)).T
    
    return coords

    

def plot_ghia_u(filename,solution_pinns, solution_fem):
    # extract solution from the ghia_u.csv file
    data = np.genfromtxt("../../ghia_u.csv")

    y = data[:,0]
    u1 = data[:,1]
    
    # calculate the error pinns vs FEM
    l2_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],2)/np.linalg.norm(solution_fem[:,1],2)
    l1_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],1)/np.linalg.norm(solution_fem[:,1],1)
    l_inf_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],np.inf)/np.linalg.norm(solution_fem[:,1],np.inf)
    
    # calculate the error pinns vs ghia
    l2_norm_pinns_ghia = np.linalg.norm(solution_pinns-u1,2)/np.linalg.norm(u1,2)
    l1_norm_pinns_ghia = np.linalg.norm(solution_pinns-u1,1)/np.linalg.norm(u1,1)
    l_inf_norm_pinns_ghia = np.linalg.norm(solution_pinns-u1,np.inf)/np.linalg.norm(u1,np.inf)
    
    # merge all errors into a list
    errors = [l2_norm_pinns_fem,l1_norm_pinns_fem,l_inf_norm_pinns_fem,l2_norm_pinns_ghia,l1_norm_pinns_ghia,l_inf_norm_pinns_ghia]
    
    
    plt.figure(figsize=(6.4,4.8))
    plt.plot(y,u1,'k-',label='Ghia et al.')
    plt.plot(y,solution_pinns,'r--',label='PINN')
    plt.plot(solution_fem[:,0],solution_fem[:,1],'b-.',label='FEM')
    plt.title('Velocity in x-direction')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('u')
    plt.savefig(f"{filename}.png",dpi=300)
    
    ## print errors
    print(f"l2_norm_pinns_fem_u = {l2_norm_pinns_fem}")
    print(f"l1_norm_pinns_fem_u = {l1_norm_pinns_fem}")
    print(f"l_inf_norm_pinns_fem_u = {l_inf_norm_pinns_fem}")
    print(f"l2_norm_pinns_ghia_u = {l2_norm_pinns_ghia}")
    print(f"l1_norm_pinns_ghia_u = {l1_norm_pinns_ghia}")
    print(f"l_inf_norm_pinns_ghia_u = {l_inf_norm_pinns_ghia}")
    
    
    
    return errors

def plot_ghia_v(filename,solution_pinns, solution_fem):
    # extract solution from the ghia_v.csv file
    data = np.genfromtxt("../../ghia_v.csv")

    x = data[:,0]
    v1 = data[:,1]
    
    # calculate the error pinns vs FEM
    l2_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],2)/np.linalg.norm(solution_fem[:,1],2)
    l1_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],1)/np.linalg.norm(solution_fem[:,1],1)
    l_inf_norm_pinns_fem = np.linalg.norm(solution_pinns-solution_fem[:,1],np.inf)/np.linalg.norm(solution_fem[:,1],np.inf)
    
    # calculate the error pinns vs ghia
    l2_norm_pinns_ghia = np.linalg.norm(solution_pinns-v1,2)/np.linalg.norm(v1,2)
    l1_norm_pinns_ghia = np.linalg.norm(solution_pinns-v1,1)/np.linalg.norm(v1,1)
    l_inf_norm_pinns_ghia = np.linalg.norm(solution_pinns-v1,np.inf)/np.linalg.norm(v1,np.inf)
    
    # merge all errors into a list
    errors = [l2_norm_pinns_fem,l1_norm_pinns_fem,l_inf_norm_pinns_fem,l2_norm_pinns_ghia,l1_norm_pinns_ghia,l_inf_norm_pinns_ghia]
    
    
    plt.figure(figsize=(6.4,4.8))
    plt.plot(x,v1,'k-',label='Ghia et al.')
    plt.plot(x,solution_pinns,'r--',label='PINN')
    plt.plot(solution_fem[:,0],solution_fem[:,1],'b-.',label='FEM')
    plt.title('Velocity in y-direction')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('v')
    plt.savefig(f"{filename}.png",dpi=300)
    
    ## print errors
    print(f"l2_norm_pinns_fem_u = {l2_norm_pinns_fem}")
    print(f"l1_norm_pinns_fem_u = {l1_norm_pinns_fem}")
    print(f"l_inf_norm_pinns_fem_u = {l_inf_norm_pinns_fem}")
    print(f"l2_norm_pinns_ghia_u = {l2_norm_pinns_ghia}")
    print(f"l1_norm_pinns_ghia_u = {l1_norm_pinns_ghia}")
    print(f"l_inf_norm_pinns_ghia_u = {l_inf_norm_pinns_ghia}")
    
    return errors
    
    
    
    
    