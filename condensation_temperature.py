import numpy as np

def get_tc(species_id):
    # return 50% solar system gas condensation temperature from Table 8 of Lodders (2003)
    try:
        tc = tc_map[species_id]
    except:
        print("Species not recognized.")    
        return None
    return tc

tc_map = {
          'Li' :   1142.0,
          'Be' :   1452.0,
          'B'  :    908.0,
          'C'  :     40.0,
          'CH'  :     40.0,
          'N'  :    123.0,
          'O'  :    180.0,
          'F'  :    734.0,
          'Na' :    958.0,
          'Mg' :   1336.0,
          'Al' :   1653.0,
          'Si' :   1310.0,
          'P'  :   1229.0,
          'S'  :    664.0,
          'K'  :   1006.0,
          'Ca' :   1517.0,
          'Sc' :   1659.0,
          'Ti' :   1582.0,
          'V'  :   1429.0,
          'Cr' :   1296.0,
          'Mn' :   1158.0,
          'Fe' :   1334.0,
          'Co' :   1352.0,
          'Ni' :   1353.0,
          'Cu' :   1037.0,
          'Zn' :    726.0,
          'Rb' :    800.0,
          'Sr' :   1464.0,
          'Sr':   1464.0,
          'Y' :   1659.0, 
          'Zr':   1741.0,
          'Ba':   1455.0,
          'La':   1578.0,
          'Ce':   1478.0,
          'Pr':   1582.0,
          'Nd':   1602.0,
          'Sm':   1590.0,
          'Eu':   1356.0,
          'Gd':   1659.0,
          'Dy':   1659.0 
          }