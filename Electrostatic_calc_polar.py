#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

###############################################
# This script is mainly used to calculate electrostatic energy between
# interested solute atoms and surronding solvent atoms. This script is
# also possible to extract several closest solvent molecules based on three different
# criterions: 1) based on total electrostatic E between the solvent molecule and 
# the interested solute atom; 2) based on the maximimum absolute electrostatic E between
# any atom within the same solvent molecule and the interested solute atom, for example,
# the solvent has 13 atoms, the script will calculate the absolute electrostatic E between
# from 1st to 13th atom inside this solvent and the interested solute atom, then use the maximum
# as the "label" for this solvent, then extract solvent molecule based on this "label";
# 3) based on the distance between the solvent molecule and the interested solute atom.
# This kind of analysis would be helpful to see clearly from electrostatic perspective, which
# solvent molecules interact well with the interested solute parts (usually the most polar parts).
# It's possible to set the whole solute for calculation as well. The generated output files would 
# have both xyz and g16 input file format, which can be used for other later calculations.
#
# You may argue cpptraj can do similar things to calculate electrostatic E, but it usually used the
# charges from your simulation parameter file, like .prmtop file if using Amber. Those charges are not NPA
# charges we commonly use in Quantum Mechanical level analysis. This script is able to use your calculated
# NPA charges for analysis. Although the charges are different from the rougly estimated ones used in simulation,
# the equation to calculate electrostatic E is the same based on my search of cpptraj manual, which considered
# the cutoff-distance (or reference distance) as well.
###############################################

# Function to calculate distance between two points
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Function to calculate electrostatic energy
def calculate_energy(qi, qj, rij, rf):
    if rij < rf:
        return qi * qj * (1 - (rij**2 / rf**2))**2 / rij
    else:
        return 0

def main(npa_file, g16_crd):
    with open(npa_file, 'r') as file:
        # Read all lines from the g16 NPA calc output file
        lines = file.readlines()

    # Initialize variables to track the positions of interest
    target_lines = []
    capture_data = False
    natural_count = 0
    target_natural_count = 4  # Change this to the desired occurrence of "Natural"
    column_names = []

    # Iterate over the lines with their index
    for index, line in enumerate(lines):
        # Check if the line contains the string "Natural"
        if "Natural" in line:
            natural_count += 1
            # Check if we've reached the desired occurrence of "Natural"
            if natural_count == target_natural_count:
                # The index of the "Natural" line is followed by "Charge" line
                charge_line_index = index + 1
                target_line_index = index + 3
                capture_data = True

        # If capture_data is True, start collecting the target lines
        if capture_data:
            # Check if the line contains only "=" characters
            if "=" in line.strip():
                break
            # Capture column names from the "Charge" line
            if index == charge_line_index:
                column_names = line.strip().split()
            # Collect the target line data which is 3 lines after "Natural"
            if index >= target_line_index:
                target_lines.append(line.strip())

    # Create a list to hold the split data
    data = []

    # Split each target line into columns and append to data list
    for line in target_lines:
        columns = line.split()
        data.append(columns)

    # Convert the data list to a pandas DataFrame with the captured column names
    df = pd.DataFrame(data, columns=column_names)

    # Filter the DataFrame to include only the columns "Atom" and "Charge"
    filtered_df = df[['Atom', 'Charge']]

    # Open the g16 input file for reading
    with open(g16_crd, 'r') as file:
        # Read all lines from the file
        lines_inp = file.readlines()

    # Initialize a list to store the g16 input crd data
    data_inp = []

    # Iterate over the lines starting from the 8th line (index 7)
    # This number depends on the g16 input file format, adjust by case
    for line in lines_inp[7:]:
        # Stop processing if a blank line is encountered
        if line.strip() == '':
            break
        # Split the line into columns based on whitespace
        columns_inp = line.split()
        # Append the columns to the data list
        data_inp.append(columns_inp)

    # Convert the data list to a pandas DataFrame with the specified column names
    df_inp = pd.DataFrame(data_inp, columns=["Atom", "X", "Y", "Z"])

    df_all = df_inp
    # append the NPA charge values read from g16 output file on the inp crd
    df_all['Charge'] = filtered_df['Charge']
    # make sure all the values become float type for later calc
    df_all['X'] = df_all['X'].astype(float)
    df_all['Y'] = df_all['Y'].astype(float)
    df_all['Z'] = df_all['Z'].astype(float)
    df_all['Charge'] = df_all['Charge'].astype(float)

    # number of solute atoms
    atm_qm = 60
    # number of solvent atoms
    atm_solvt = 13
    # Number of closest solvent molecules to find
    num_closest_solvents = 3
    # Define the cutoff distance for electrostatic E calc
    cutoff_distance = 12.0

    # adjust above parameters based on user's need
    length = len(sys.argv)
    if (length > 1):
            for k in range(length):
                if(sys.argv[k] == '-aq'):
                    atm_qm = int(sys.argv[k+1])
                elif(sys.argv[k] == '-as'):
                    atm_solvt = int(sys.argv[k+1])
                elif(sys.argv[k] == '-ncs'):
                    num_closest_solvents = int(sys.argv[k+1])
                elif(sys.argv[k] == '-cd'):
                    cutoff_distance = float(sys.argv[k+1])

    # dataframe for solute
    df_qm = df_all.iloc[:atm_qm]
    # dataframe for solvent
    df_mm = df_all.iloc[atm_qm:]
    df_mm = df_mm.reset_index(drop=True)

    # Define the specified solute atoms (convert to zero-based index)
    # which user wants to calc electrostatic E on
    qm_atoms_to_calculate = [0, 1, 2, 18, 22, 28] 

    # Initialize a dictionary to store the results for each specified solute atom
    energies = {atom: 0 for atom in qm_atoms_to_calculate}

    # Loop over each specified solute atom
    for qm_atom in qm_atoms_to_calculate:
        qi = df_qm.loc[qm_atom, 'Charge']
        x1, y1, z1 = df_qm.loc[qm_atom, ['X', 'Y', 'Z']]

        # Loop over each solvent atom
        for mm_atom in df_mm.index:
            qj = df_mm.loc[mm_atom, 'Charge']
            x2, y2, z2 = df_mm.loc[mm_atom, ['X', 'Y', 'Z']]

            # Calculate distance
            rij = calculate_distance(x1, y1, z1, x2, y2, z2)

            # Calculate energy
            energy = calculate_energy(qi, qj, rij, cutoff_distance)

            # Add to the total energy for this solute atom
            energies[qm_atom] += energy

    # Write the electrostatic energy to a new .txt file
    with open('electrostatic_energy.txt', 'w') as file:
        # Write the header
        file.write("Atom\tElectrostatic Energy\n")

        # Write the electrostatic energy for each specified solute atom
        for atom in qm_atoms_to_calculate:
            atom_type = df_qm.loc[atom, 'Atom']
            energy = energies[atom]
            file.write(f"{atom_type}\t{energy}\n")

    # Criterion 1: Sum of Electrostatic Energy Over All Atoms in the Solvent Molecule
    closest_solvent_indices_sum = {atom: [] for atom in qm_atoms_to_calculate}

    # Loop over each specified solute atom
    for qm_atom in qm_atoms_to_calculate:
        qi = df_qm.loc[qm_atom, 'Charge']
        x1, y1, z1 = df_qm.loc[qm_atom, ['X', 'Y', 'Z']]

        # List to store total energy and corresponding solvent start indices
        solvent_energies = []

        # Loop over each solvent molecule (each molecule has 13 atoms)
        for start_idx in range(0, len(df_mm), atm_solvt):
            total_energy = 0
            for mm_atom in range(start_idx, start_idx + atm_solvt):
                qj = df_mm.loc[mm_atom, 'Charge']
                x2, y2, z2 = df_mm.loc[mm_atom, ['X', 'Y', 'Z']]
                rij = calculate_distance(x1, y1, z1, x2, y2, z2)
                total_energy += calculate_energy(qi, qj, rij, cutoff_distance)

            # Append the total energy and corresponding solvent start index to the list
            solvent_energies.append((total_energy, start_idx))

        # Sort the solvent energies and get the indices of the closest solvents
        solvent_energies.sort(key=lambda x: abs(x[0]),reverse=True)  # Sort by total energy in descending order
        closest_solvents = [idx for _, idx in solvent_energies[:num_closest_solvents]]
        closest_solvent_indices_sum[qm_atom] = closest_solvents

    # Collect unique closest solvent indices
    unique_solvent_indices_sum = set()
    for solvents in closest_solvent_indices_sum.values():
        unique_solvent_indices_sum.update(solvents)

    # Prepare the data for the XYZ file
    xyz_data_sum = []

    # Add solute atoms to the XYZ data
    for idx in df_qm.index:
        atom_type = df_qm.loc[idx, 'Atom']
        x, y, z = df_qm.loc[idx, ['X', 'Y', 'Z']]
        xyz_data_sum.append((atom_type, x, y, z))

    # Add unique closest solvent molecules to the XYZ data
    for start_idx in unique_solvent_indices_sum:
        for idx in range(start_idx, start_idx + atm_solvt):
            atom_type = df_mm.loc[idx, 'Atom']
            x, y, z = df_mm.loc[idx, ['X', 'Y', 'Z']]
            xyz_data_sum.append((atom_type, x, y, z))

    # Write solute plus solvent data to an XYZ/g16 formatted file
    with open('extract_total_E.xyz', 'w') as file:
        # Write the total number of atoms
        file.write(f"{len(xyz_data_sum)}\n")
        # Write a blank line
        file.write("\n")
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_sum:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    with open('extract_total_E.com', 'w') as file:
        # Write the total number of atoms
        file.write("# \n")
        # Write a blank line
        file.write("\n")
        file.write('Title\n')
        file.write('\n')
        file.write('0 1\n')
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_sum:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    # Criterion 2: Maximum Absolute Contribution of Electrostatic Energy from Any Atom in the Solvent Molecule
    closest_solvent_indices_max = {atom: [] for atom in qm_atoms_to_calculate}

    # Loop over each specified QM atom
    for qm_atom in qm_atoms_to_calculate:
        qi = df_qm.loc[qm_atom, 'Charge']
        x1, y1, z1 = df_qm.loc[qm_atom, ['X', 'Y', 'Z']]

        # List to store max absolute energy and corresponding solvent start indices
        solvent_energies = []

        # Loop over each MM solvent molecule (each molecule has 13 atoms)
        for start_idx in range(0, len(df_mm), atm_solvt):
            max_energy = 0
            for mm_atom in range(start_idx, start_idx + atm_solvt):
                qj = df_mm.loc[mm_atom, 'Charge']
                x2, y2, z2 = df_mm.loc[mm_atom, ['X', 'Y', 'Z']]
                rij = calculate_distance(x1, y1, z1, x2, y2, z2)
                energy = calculate_energy(qi, qj, rij, cutoff_distance)
                if abs(energy) > abs(max_energy):
                    max_energy = energy

            # Append the max energy and corresponding solvent start index to the list
            solvent_energies.append((max_energy, start_idx))

        # Sort the solvent energies and get the indices of the closest solvents
        solvent_energies.sort(key=lambda x: abs(x[0]), reverse=True)  # Sort by absolute energy in descending order
        closest_solvents = [idx for _, idx in solvent_energies[:num_closest_solvents]]
        closest_solvent_indices_max[qm_atom] = closest_solvents

    # Collect unique closest solvent indices
    unique_solvent_indices_max = set()
    for solvents in closest_solvent_indices_max.values():
        unique_solvent_indices_max.update(solvents)

    # Prepare the data for the XYZ file
    xyz_data_max = []

    # Add solute atoms to the XYZ data
    for idx in df_qm.index:
        atom_type = df_qm.loc[idx, 'Atom']
        x, y, z = df_qm.loc[idx, ['X', 'Y', 'Z']]
        xyz_data_max.append((atom_type, x, y, z))

    # Add unique closest solvent molecules to the XYZ data
    for start_idx in unique_solvent_indices_max:
        for idx in range(start_idx, start_idx + atm_solvt):
            atom_type = df_mm.loc[idx, 'Atom']
            x, y, z = df_mm.loc[idx, ['X', 'Y', 'Z']]
            xyz_data_max.append((atom_type, x, y, z))

    # Write solute plus solvent data to an XYZ/g16 formatted file
    with open('extract_absmax_E.xyz', 'w') as file:
        # Write the total number of atoms
        file.write(f"{len(xyz_data_max)}\n")
        # Write a blank line
        file.write("\n")
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_max:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    with open('extract_absmax_E.com', 'w') as file:
        # Write the total number of atoms
        file.write("# \n")
        # Write a blank line
        file.write("\n")
        file.write('Title\n')
        file.write('\n')
        file.write('0 1\n')
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_max:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    # Criterion 3: Minimum distance between QM atoms and solvent molecules
    closest_solvent_indices_dist = {atom: [] for atom in qm_atoms_to_calculate}

    # Loop over each specified QM atom
    for qm_atom in qm_atoms_to_calculate:
        x1, y1, z1 = df_qm.loc[qm_atom, ['X', 'Y', 'Z']]

        # List to store distances and corresponding solvent start indices
        solvent_distances = []

        # Loop over each MM solvent molecule (each molecule has 13 atoms)
        for start_idx in range(0, len(df_mm), atm_solvt):
            # Calculate the minimum distance between the QM atom and the atoms in the solvent molecule
            min_distance = float('inf')
            for mm_atom in range(start_idx, start_idx + atm_solvt):
                x2, y2, z2 = df_mm.loc[mm_atom, ['X', 'Y', 'Z']]
                distance = calculate_distance(x1, y1, z1, x2, y2, z2)
                if distance < min_distance:
                    min_distance = distance

            # Append the minimum distance and corresponding solvent start index to the list
            solvent_distances.append((min_distance, start_idx))

        # Sort the solvent distances and get the indices of the closest solvents
        solvent_distances.sort()
        closest_solvents = [idx for _, idx in solvent_distances[:num_closest_solvents]]
        closest_solvent_indices_dist[qm_atom] = closest_solvents

    # Collect unique closest solvent indices
    unique_solvent_indices_dist = set()
    for solvents in closest_solvent_indices_dist.values():
        unique_solvent_indices_dist.update(solvents)

    # Prepare the data for the XYZ file
    xyz_data_dist = []

    # Add solute atoms to the XYZ data
    for idx in df_qm.index:
        atom_type = df_qm.loc[idx, 'Atom']
        x, y, z = df_qm.loc[idx, ['X', 'Y', 'Z']]
        xyz_data_dist.append((atom_type, x, y, z))

    # Add unique closest solvent molecules to the XYZ data
    for start_idx in unique_solvent_indices_dist:
        for idx in range(start_idx, start_idx + atm_solvt):
            atom_type = df_mm.loc[idx, 'Atom']
            x, y, z = df_mm.loc[idx, ['X', 'Y', 'Z']]
            xyz_data_dist.append((atom_type, x, y, z))

    # Write solute plus solvent data to an XYZ/g16 formatted file
    with open('extract_mindist.xyz', 'w') as file:
        # Write the total number of atoms
        file.write(f"{len(xyz_data_dist)}\n")
        # Write a blank line
        file.write("\n")
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_dist:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

    with open('extract_mindist.com', 'w') as file:
        # Write the total number of atoms
        file.write("# \n")
        # Write a blank line
        file.write("\n")
        file.write('Title\n')
        file.write('\n')
        file.write('0 1\n')
        # Write the atom type and coordinates
        for atom_type, x, y, z in xyz_data_dist:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

if __name__ == "__main__":
    npa_file = 'total_top.out' # Replace with your actual file path for g16 output containing NPA charges
    g16_crd = 'total_top.com' # Replace with actual file path for g16 input file used to calc NPA charges
    main(npa_file, g16_crd)
