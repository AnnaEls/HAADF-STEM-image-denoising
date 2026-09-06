import atomap.api as am
import hyperspy.api as hs

def find_atoms(image, separation, plot = False):
  atom_positions = am.get_atom_positions(image, separation=separation)
  sublattice = am.Sublattice(atom_positions, image=image.data)
  sublattice.find_nearest_neighbors()
  sublattice.refine_atom_positions_using_center_of_mass()
  sublattice.refine_atom_positions_using_2d_gaussian()
  if plot:
    sublattice.plot()
  return sublattice
