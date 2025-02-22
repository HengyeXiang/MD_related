Some scripts can be useful for file generation or energy calculation after the Molecular Dynamics runs.<br>
1) extract closest solvent molecules around solute based on a specified distance or number threshold:<br>
   ```python SolvtAbsOM_gview.py -as 13 -td 15.0 -tn 59``` <br>
   Here ```-as``` refers to number of solvent atoms, ```-td``` refers to threshold distance, ```-tn``` refers to threshold number. <br>
   
2) extract closest solvent molecules around interested solute parts (e.g. polar parts) based on electrostatic energy calculated using NPA charges:<br>
   ```python Electrostatic_calc_polar.py -as 13 -ncs 3``` <br>
   Here ```-as``` refers to number of solvent atoms, ```-ncs``` refers to desired number you want for closest solvent molecules. <br>
   <br>
Details are provided in the comments part at the beginning of the script. <br>
Example input and output files for each script are provided as well.<br>
