import os
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd

###################################
# This script is used to capture closest solvent molecules around solute.
# This can be quite useful after you run a Molecular Dynamics simulation,
# you want to use a smaller solvation shell for high-accuracy Quantum Chmeistry
# related calculation, like DFT or ONIOM calculation inside Gaussian package.
# VMD and PyMOL tools can do similar things based on threshold distance but it 
# usually requires a really standard PDB file to operate, like you need to maintain
# the residue names, etc. Instead, this script is more general to different kinds of
# PDB files, like the one you generated from GView.
# 
# In addition to use distance as the selection threshold, this script also supports
# using a specific number of solvent molecules as a threshold. Althogh currently it uses
# the center of mass of solute as the reference for distance-related calculations,
# it's possible to change this to something else like the geometric center, only polar/nonpolar
# parts of the solute. What you need to do is just replacing the centroid in the code with another desired point.
# 
# In the script's example, it studied the result from a QM/MM AIMD simulation of an organic reaction in solvents.
# We may want to calculate solvation energy of a smaller solvation shell, the input file
# of which can be prepared using this script. It's also possible to run ONIOM calculations
# based on the smaller solvation shell. To use this script, make sure you have the solute molecule 
# at the beginning of the coordinate part of the PDB file and all solvents following that.
###################################

# function to calculate the distance between atom and center of mass
def calc_length(df,i,comass):
    row_data = df.iloc[[i], [1, 2, 3]]
    x = row_data.to_numpy(dtype=np.float64)
    y = np.array(comass,dtype=np.float64)
    sum_d = 0.0
    for v in range(len(y)):
        sum_d += (x[0][v]-y[v])**2
    distance = math.sqrt(sum_d)
    return round(distance,4)

# function to read the relevant atomic data into a DataFrame
def read_pdb_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # for the gview generated PDB file, coordinates start from the 3rd line
        # possible to replace the value 2 below to other int based on input format
        for line in lines[2:]:
            # stop reading at the line starts with END mark
            # if the PDB file doesn't have END mark line, remove this if part
            if line.startswith("END"):
                break
            parts = line.split()
            x = float(parts[4])
            y = float(parts[5])
            z = float(parts[6])
            element = parts[-1]
            data.append([element, x, y, z])
    return pd.DataFrame(data, columns=['Element', 'X', 'Y', 'Z'])

# function to calculate the coordinates for center of mass
def calculate_mass_centroid(df, atm_qm):
    atomic_masses = {'H': 1.008, 'C': 12.011, 'O': 15.999, 'P': 30.974}
    df['Mass'] = df['Element'].map(atomic_masses)
    subset = df.head(atm_qm)
    total_mass = subset['Mass'].sum()
    centroid_x = (subset['X'] * subset['Mass']).sum() / total_mass
    centroid_y = (subset['Y'] * subset['Mass']).sum() / total_mass
    centroid_z = (subset['Z'] * subset['Mass']).sum() / total_mass
    return centroid_x, centroid_y, centroid_z

# function to write xyz file from single dataframe
def write_xyz(file_path, df):
    with open(file_path, 'w') as file:
        file.write(f"{len(df)}\n")
        file.write("\n")
        for index, row in df.iterrows():
            file.write(f"{row['Element']} {row['X']} {row['Y']} {row['Z']}\n")
            
# function to write g16 input file from single dataframe
def write_single_gcrt(file_path, df):
    with open(file_path, 'w') as file:
        file.write("# m062x/6-311+g(d,p)\n")
        file.write("\n")
        file.write('Title\n')
        file.write('\n')
        file.write('0 1\n')
        for index, row in df.iterrows():
            file.write(f"{row['Element']} {row['X']} {row['Y']} {row['Z']}\n")
        file.write("\n")
        file.write("\n")

# function to write g16 input file from two dataframes (combining their coordinates)        
def write_comb_gcrt(file_path, df1, df2):
    with open(file_path, 'w') as file:
        file.write("# m062x/6-311+g(d,p)\n")
        file.write("\n")
        file.write('Title\n')
        file.write('\n')
        file.write('0 1\n')
        for index, row in df1.iterrows():
            file.write(f"{row['Element']} {row['X']} {row['Y']} {row['Z']}\n")
        for index, row in df2.iterrows():
            file.write(f"{row['Element']} {row['X']} {row['Y']} {row['Z']}\n")
        file.write("\n")
        file.write("\n")

        
def main(path_pdb, folder_name):
    df_all = read_pdb_to_dataframe(path_pdb)
    
    # total number of solute atoms treated with QM approach in MD
    atm_qm = 60
    # number of solvent atoms
    atm_solvt = 13
    # threshold distance for picking solvents
    trsd_dist = 15.0
    # threshold number of solvents to pick
    trsd_num = 59
    # number of aldehyde atoms (one part of the solute complex)
    num_adh = 14
    # number of ylide atoms (the other part of solute complex)
    num_yld = int(int(atm_qm) - int(num_adh))

    # change above relevant parameters based on user specification
    length = len(sys.argv)
    if (length > 1):
            for k in range(length):
                if(sys.argv[k] == '-aq'):
                    atm_qm = int(sys.argv[k+1])
                elif(sys.argv[k] == '-as'):
                    atm_solvt = int(sys.argv[k+1])
                elif(sys.argv[k] == '-td'):
                    trsd_dist = float(sys.argv[k+1])
                elif(sys.argv[k] == '-tn'):
                    trsd_num = int(sys.argv[k+1])

    # calculate the center of mass for solute complex
    qm_cent = calculate_mass_centroid(df_all, atm_qm)

    # make sure the X,Y,Z column in the dataframe have the float data type
    df_all['X'] = df_all['X'].astype(float)
    df_all['Y'] = df_all['Y'].astype(float)
    df_all['Z'] = df_all['Z'].astype(float)

    # dataframe for aldehyde part of solute complex
    df_adh = df_all.iloc[:num_adh]
    # dataframe for ylide part of solute complex
    df_yld = df_all.iloc[num_adh:atm_qm]
    # dataframe for the whole solute complex
    df_qm = df_all.iloc[:atm_qm]
    # dataframe for all solvent molecules
    df_mm = df_all.iloc[atm_qm:]
    df_mm = df_mm.reset_index(drop=True)
    
    # calculate the distance between all atoms in solvents and center of mass for solute
    for i in range(len(df_mm)):
        df_mm.loc[i, 'Dist'] = calc_length(df_mm,i,qm_cent)

    # Extract solvents based on threshold distance, default is 15 Angstrom
    results = []
    for i in range(0, len(df_mm), atm_solvt):
        if df_mm.loc[i:i+atm_solvt-1, 'Dist'].min() <= trsd_dist:
            results.append(df_mm.loc[i:i+atm_solvt-1, ['Element', 'X', 'Y', 'Z']])
    filtered_df = pd.concat(results, ignore_index=True)
    print('Number of solvent molecules within the distance threshold: ', int(len(filtered_df)/atm_solvt))

    # Extract solvents based on threshold number, default is based on density calc.
    min_dist = []
    for i in range(0, len(df_mm), atm_solvt):
        distance = df_mm.loc[i:i+atm_solvt-1, 'Dist'].min()
        min_dist.append((i, distance))
    min_dist.sort(key=lambda x: x[1])
    top_indices = [index for index, _ in min_dist[:trsd_num]]
    top_mols = pd.concat([df_mm.loc[idx:idx+atm_solvt-1, ['Element', 'X', 'Y', 'Z']] for idx in top_indices],
                    ignore_index=True)
    
    # check whether the folder already exists
    os.makedirs(folder_name, exist_ok=True)
    # assign output xyz file paths
    path_mm_within = os.path.join(folder_name, 'solvent.xyz')
    path_ald = os.path.join(folder_name, 'aldehyde.xyz')
    path_yld = os.path.join(folder_name, 'ylide.xyz')
    path_mm_top = os.path.join(folder_name, 'solvent_top.xyz')
    # assign output g16 input file paths
    path_qm = os.path.join(folder_name, 'ts.com')
    path_solvt_top = os.path.join(folder_name, 'solvent_top.com')
    path_total_top = os.path.join(folder_name, 'total_top.com')

    # write xyz files
    # xyz file for solvents within a specified distance
    write_xyz(path_mm_within, filtered_df)
    write_xyz(path_ald, df_adh)
    write_xyz(path_yld, df_yld)
    # xyz file for solvents with a specified number
    write_xyz(path_mm_top, top_mols)
    # write g16 input files
    write_single_gcrt(path_qm, df_qm)
    # g16 input file for solvents with a specified number
    write_single_gcrt(path_solvt_top, top_mols)
    write_comb_gcrt(path_total_top, df_qm, top_mols)

if __name__ == "__main__":
    path_pdb = glob.glob('./*.pdb')[0]  # Find the PDB file under current directory
    folder_name = "TS_SolvtAbs"  # Replace with desired output folder name
    main(path_pdb, folder_name)    
