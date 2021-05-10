from mpoles_for_fmm import *
import numpy as np
import scipy as sp
from scipy.special import comb
from math import factorial
from mpoles_mk2 import *
import time

# PointCharge Class
class PointCharge:

    # q is in electrons, x, y, z are in bohrs
    def __init__(self, q, x, y, z):
        self.q = q
        self.x = x
        self.y = y
        self.z = z

    # Distance is calculated in bohrs
    def calc_distance(self, pc):
        r12 = [self.x - pc.x, self.y - pc.y, self.z - pc.z]
        return np.linalg.norm(r12)

    # Energy is in Hartrees
    def calc_interaction_energy(self, pc):
        return self.q * pc.q / self.calc_distance(pc)
    
# PairwisePointChargeSystem Class
class PairwisePointChargeSystem:

    # Calculates the interaction energy for a bunch of point charges
    def __init__(self, point_charges):
        self.point_charges = point_charges
        self.interaction_energy = 0.0

        for i in range(len(self.point_charges)):
            for j in range(i+1, len(self.point_charges)):
                self.interaction_energy += (self.point_charges[i]).calc_interaction_energy(self.point_charges[j])


    def get_energy(self):
        return self.interaction_energy

# FMMBox Class
class FMMBox:

    def __init__(self, tree, treeindex, parent, level, lmax, ws, charges, r0, length):
        self.tree = tree
        self.treeindex = treeindex
        self.parent = parent
        self.level = level
        self.lmax = lmax
        self.ws = ws
        self.charges = charges
        self.r0 = r0
        self.length = length

        # Center of the Box
        self.rC = np.array([r0[0]+0.5*length, r0[1]+0.5*length, r0[2]+0.5*length])

        self.nf_thresh = self.ws * self.length

        # Multipoles, Vff vector, and interaction energy
        self.multipoles = RealRegularHarmonics(self.rC, self.lmax)
        self.Vff = RealRegularHarmonics(self.rC, self.lmax)
        self.energy = 0.0
        
        self.children = {}
        self.children['000'] = None
        self.children['100'] = None
        self.children['010'] = None
        self.children['001'] = None
        self.children['110'] = None
        self.children['011'] = None
        self.children['101'] = None
        self.children['111'] = None

        # Near Field
        self.nf = []
        # Local Far Field
        self.lff = []
    
    # Set Near Field and Local Far Field Boxes
    def set_nf_lff(self):

        if (len(self.charges) == 0):
            return

        if self.parent != None:
            for ind, box in self.parent.children.items():
                if box == self:
                    continue
                rC = box.rC
                if (np.linalg.norm(self.rC - rC) <= self.nf_thresh * np.sqrt(3)):
                    self.nf.append(box)
                else:
                    self.lff.append(box)

            for box in self.parent.nf:
                for ind, child in box.children.items():
                    rC = child.rC
                    if (np.linalg.norm(self.rC - rC) <= self.nf_thresh * np.sqrt(3)):
                        self.nf.append(child)
                    else:
                        self.lff.append(child)

        
    def add_charge(self, charge):
        self.charges.append(charge)

    # Adds another level of children for to the FMM Tree Structure
    def make_children(self):

        half = 0.5 * self.length
        r0 = self.r0
        dx = np.array([half, 0.0, 0.0])
        dy = np.array([0.0, half, 0.0])
        dz = np.array([0.0, 0.0, half])
        
        ti = self.treeindex

        self.children['000'] = FMMBox(self.tree, (2*ti[0], 2*ti[1], 2*ti[2]), self, self.level+1, self.lmax, self.ws, [], r0, half)
        self.children['100'] = FMMBox(self.tree, (2*ti[0]+1, 2*ti[1], 2*ti[2]), self, self.level+1, self.lmax, self.ws, [], r0+dx, half)
        self.children['010'] = FMMBox(self.tree, (2*ti[0], 2*ti[1]+1, 2*ti[2]), self, self.level+1, self.lmax, self.ws, [], r0+dy, half)
        self.children['001'] = FMMBox(self.tree, (2*ti[0], 2*ti[1], 2*ti[2]+1), self, self.level+1, self.lmax, self.ws, [], r0+dz, half)
        self.children['110'] = FMMBox(self.tree, (2*ti[0]+1, 2*ti[1]+1, 2*ti[2]), self, self.level+1, self.lmax, self.ws, [], r0+dx+dy, half)
        self.children['011'] = FMMBox(self.tree, (2*ti[0], 2*ti[1]+1, 2*ti[2]+1), self, self.level+1, self.lmax, self.ws, [], r0+dy+dz, half)
        self.children['101'] = FMMBox(self.tree, (2*ti[0]+1, 2*ti[1], 2*ti[2]+1), self, self.level+1, self.lmax, self.ws, [], r0+dx+dz, half)
        self.children['111'] = FMMBox(self.tree, (2*ti[0]+1, 2*ti[1]+1, 2*ti[2]+1), self, self.level+1, self.lmax, self.ws, [], r0+dx+dy+dz, half)

        x1 = self.rC[0]
        y1 = self.rC[1]
        z1 = self.rC[2]

        for i in range(len(self.charges)):
            qx = self.charges[i].x
            qy = self.charges[i].y
            qz = self.charges[i].z

            if (qx < x1 and qy < y1 and qz < z1):
                self.children['000'].add_charge(self.charges[i])
            elif (qx >= x1 and qy < y1 and qz < z1):
                self.children['100'].add_charge(self.charges[i])
            elif (qx < x1 and qy >= y1 and qz < z1):
                self.children['010'].add_charge(self.charges[i])
            elif (qx < x1 and qy < y1 and qz >= z1):
                self.children['001'].add_charge(self.charges[i])
            elif (qx >= x1 and qy >= y1 and qz < z1):
                self.children['110'].add_charge(self.charges[i])
            elif (qx < x1 and qy >= y1 and qz >= z1):
                self.children['011'].add_charge(self.charges[i])
            elif (qx >= x1 and qy < y1 and qz >= z1):
                self.children['101'].add_charge(self.charges[i])
            elif (qx >= x1 and qy >= y1 and qz >= z1):
                self.children['111'].add_charge(self.charges[i])


    # Calculates regular spherical multipoles; 0 = monopole, 1 = dipole, 
    # 2 = quadrupole, 3 = octupole, etc... (regular solid harmonics)
    def calc_multipoles(self):

        for charge in self.charges:
            q = charge.q
            dx = self.rC[0] - charge.x
            dy = self.rC[1] - charge.y
            dz = self.rC[2] - charge.z

            temp = RealRegularHarmonics(self.rC, self.lmax)
            temp.compute(q, dx, dy, dz)

            self.multipoles.add(temp)

    # Calculates the multipoles of a box based on it's children's multipoles
    def calc_multipoles_from_children(self):

        for label, child in self.children.items():

            if (len(child.charges) == 0):
                continue
            
            tmpoles = child.multipoles.translate(self.rC)

            self.multipoles.add(tmpoles)
    
    # Calculate far field vector from local far-field as well as parents' far field
    def calc_far_field_vector(self):

        if (len(self.charges) == 0):
            return

        for fbox in self.lff:
            fmpole = fbox.multipoles
            self.Vff.add(fmpole.far_field_vector(self.rC))

        # Contribution from the parent's far field
        if self.parent != None:
            pff_cont = self.parent.Vff.irregular_translate(self.rC)
            self.Vff.add(pff_cont)

    # Calculate the box's interaction energy
    def calc_energy(self):

        if (len(self.charges) == 0):
            return

        self.energy += PairwisePointChargeSystem(self.charges).get_energy()

        for q1 in self.charges:
            for near in self.nf:
                for q2 in near.charges:
                    self.energy += 0.5 * q1.q * q2.q / q1.calc_distance(q2)

        for l in range(self.lmax+1):
            self.energy += 0.5 * np.dot(self.multipoles.Rlm[l], self.Vff.Rlm[l])


# FMMTree Class
class FMMTree:

    def get_dimensions(self):

        self.min_x = 0.0
        self.max_x = 0.0

        self.min_y = 0.0
        self.max_y = 0.0

        self.min_z = 0.0
        self.max_z = 0.0

        for i in range(len(self.charges)):
            charge = self.charges[i]

            qx = charge.x
            qy = charge.y
            qz = charge.z

            if (i == 0):
                self.min_x = qx
                self.max_x = qx

                self.min_y = qy
                self.max_y = qy

                self.min_z = qz
                self.max_z = qz

            else:
                if qx < self.min_x:
                    self.min_x = qx
                elif qx > self.max_x:
                    self.max_x = qx

                if qy < self.min_y:
                    self.min_y = qy
                elif qy > self.max_y:
                    self.max_y = qy
                
                if qz < self.min_z:
                    self.min_z = qz
                elif qz > self.max_z:
                    self.max_z = qz

    def __init__(self, levels, charges, lmax, ws=2):
        self.levels = levels
        self.charges = charges
        self.lmax = lmax
        self.ws = ws
        self.energy = 0.0
        
        dim = 2**levels - 1

        self.timings = {}
        self.timings["CHILD MAKER"] = 0.0
        self.timings["CALCULATE MULTIPOLES"] = 0.0
        self.timings["CALCULATE ENERGY"] = 0.0

        self.get_dimensions()
        
        length = max(self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z)
        r0 = np.array([self.min_x, self.min_y, self.min_z])

        self.root = FMMBox(tree=self, treeindex=(1,1,1), parent=None, level=0, lmax=self.lmax, ws=self.ws, charges=self.charges, r0=r0, length=length)

        start = time.time()
        self.child_maker(self.root)
        self.timings["CHILD MAKER"] += (time.time() - start)

        start = time.time()
        self.calculate_multipoles(self.root)
        self.timings["CALCULATE MULTIPOLES"] += (time.time() - start)

        start = time.time()
        self.calculate_energy(self.root)
        self.timings["CALCULATE ENERGY"] += (time.time() - start)

    # Makes children as well as set the box index array
    def child_maker(self, box):

        if box.level == self.levels - 1:
            return

        box.make_children()

        for ind in ['000', '100', '010', '001', '110', '011', '101', '111']:
            self.child_maker(box.children[ind])

    # Uses a post-order traversal to calculate multipoles at each level
    def calculate_multipoles(self, box):

        if box == None:
            return
        
        for ind in ['000', '100', '010', '001', '110', '011', '101', '111']:
            self.calculate_multipoles(box.children[ind])

        if box.level == self.levels - 1:
            box.calc_multipoles()
            return
        else:
            box.calc_multipoles_from_children()

    # Use a preorder traversal to calculate the energy of every box, as well as calculate Vff
    def calculate_energy(self, box):

        if box == None or (len(box.charges) == 0):
            return

        box.set_nf_lff()
        box.calc_far_field_vector()

        if (box.level == self.levels - 1):
            box.calc_energy()
            self.energy += box.energy
            return
        
        for ind in ['000', '100', '010', '001', '110', '011', '101', '111']:
            self.calculate_energy(box.children[ind])

    def get_energy(self):
        return self.energy