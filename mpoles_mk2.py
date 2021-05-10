import decimal
import numpy as np
from math import factorial as ft
from scipy.special import comb as cb
from mpoles_for_fmm import SphericalRotationMatrixGenerator
from mpoles_for_fmm import binomial
import time

class RealRegularHarmonics:

    # Positive Index To -l to l
    def to_signed(self, m):
        if m == 0:
            return 0
        elif (m % 2 == 0):
            return -(m//2)
        else:
            return (m+1)//2
    
    def to_unsigned(self, m):
        if m == 0:
            return 0
        elif m < 0:
            return -2*m
        else:
            return 2*m-1
    
    def kronecker(self, a, b):
        if (a == b):
            return 1
        else:
            return 0

    def __init__(self, R_a, lmax):
        # Rc and Rs are in Helgaker Convention, Rlm is in the Convention given in the Paper (J. Phys. Chem. 1996, 100, 6342-6347)
        self.Rc = [None] * (lmax+1)
        self.Rs = [None] * (lmax+1)
        self.Rlm = [None] * (lmax+1)
        self.lmax = lmax

        self.R_a = R_a

        for l in range(self.lmax+1):
            self.Rc[l] = np.zeros(l+1)
            self.Rs[l] = np.zeros(l+1)
            self.Rlm[l] = np.zeros(2*l+1)

    def add(self, rrh):
        for l in range(self.lmax+1):
            self.Rc[l] += rrh.Rc[l]
            self.Rs[l] += rrh.Rs[l]
            self.Rlm[l] += rrh.Rlm[l]

    def compute(self, q, x, y, z):

        r2 = x**2 + y**2 + z**2
        
        for l in range(self.lmax+1):

            if (l == 0):
                self.Rc[l][0] = 1.0
                self.Rs[l][0] = 0.0

            # Helgaker Equ. 9.13.78 to 9.13.82
            else:
                # l - 1 > m terms
                for m in range(l-1):
                    denom = (l+m)*(l-m)
                    self.Rc[l][m] = ((2*l-1) * z * self.Rc[l-1][m] - r2 * self.Rc[l-2][m]) / denom
                    self.Rs[l][m] = ((2*l-1) * z * self.Rs[l-1][m] - r2 * self.Rs[l-2][m]) / denom

                # m = l-1 contribution
                self.Rc[l][l-1] = z*self.Rc[l-1][l-1]
                self.Rs[l][l-1] = z*self.Rs[l-1][l-1]

                # m = l contribution
                self.Rc[l][l] = -(x*self.Rc[l-1][l-1] - y*self.Rs[l-1][l-1]) / (2*l)
                self.Rs[l][l] = -(y*self.Rc[l-1][l-1] + x*self.Rs[l-1][l-1]) / (2*l)

            # Helgaker Equ. 9.13.49
            # Odd indices (Cosine) => m = (ind+1)//2
            # Even indices (Sine) => m = -ind//2
            for mu in range(2*l+1):
                m = self.to_signed(mu)
                if (mu == 0):
                    self.Rlm[l][mu] = self.Rc[l][m]
                elif (mu % 2 == 0):
                    self.Rlm[l][mu] = self.Rs[l][-m]
                else:
                    self.Rlm[l][mu] = self.Rc[l][m]

                if m == 0:
                    prefactor = ft(l)
                else:
                    prefactor = (-1)**m * np.sqrt(2.0 * ft(l-m) * ft(l+m))

                self.Rlm[l][mu] *= prefactor

        for l in range(self.lmax+1):
            self.Rc[l] *= q
            self.Rs[l] *= q
            self.Rlm[l] *= q

    # Translation function for irregular harmonics
    def irregular_translate(self, R_b):
        transR = RealRegularHarmonics(R_b, self.lmax)
        R_ab = R_b - self.R_a
        R = np.linalg.norm(R_ab)
        dmats = SphericalRotationMatrixGenerator(self.R_a, R_b)

        rot_mpoles = []

        # Rotate Multipoles to direction of translation
        for l in range(0, self.lmax+1):
            rot_mpoles.append(np.array(np.dot(dmats.D(l), self.Rlm[l])))

        # Translate rotated multipoles
        for l in range(0, self.lmax+1):
            for j in range(l, self.lmax+1):
                for m in range(-l, l+1):

                    if m <= 0:
                        mu = 2*(-m)
                    else:
                        mu = 2*m-1

                    absm = abs(m)

                    coef = np.sqrt(cb(j+m,l+m) * cb(j-m,l-m))

                    transR.Rlm[l][mu] += coef * (-R)**(j-l) * rot_mpoles[j][mu]

        # Back-Rotation of Multipoles and set Rc and Rs
        for l in range(0, self.lmax+1):
            transR.Rlm[l] = np.dot(dmats.D(l).T, transR.Rlm[l])

        return transR
    
    # Return the translate version of this multipole on another center
    def translate(self, R_b):
        transR = RealRegularHarmonics(R_b, self.lmax)
        R_ab = R_b - self.R_a
        R = np.linalg.norm(R_ab)
        dmats = SphericalRotationMatrixGenerator(self.R_a, R_b)

        rot_mpoles = []

        # Rotate Multipoles to direction of translation
        for l in range(0, self.lmax+1):
            rot_mpoles.append(np.array(np.dot(dmats.D(l), self.Rlm[l])))

        # Translate rotated multipoles
        for l in range(0, self.lmax+1):
            for j in range(0, l+1):
                for m in range(-j, j+1):

                    if m <= 0:
                        mu = 2*(-m)
                    else:
                        mu = 2*m-1

                    absm = abs(m)

                    coef = np.sqrt(cb(l+m,j+m) * cb(l-m,j-m))

                    transR.Rlm[l][mu] += coef * (-R)**(l-j) * rot_mpoles[j][mu]

        # Back-Rotation of Multipoles and set Rc and Rs
        for l in range(0, self.lmax+1):
            transR.Rlm[l] = np.dot(dmats.D(l).T, transR.Rlm[l])
        
        return transR

    def mvals(self, l):
        ms = [0]
        for m in range(1,l+1):
            ms.extend([m,-m])
        return ms

    def build_T_spherical(self, la, lb, R):
        ms = self.mvals(min(la,lb))
        return np.array([((-1)**(lb-m)) * np.sqrt(1.0 * binomial(la+lb, la+m) * binomial(la+lb, la-m)) for m in ms]) / R**(la+lb+1)
    
    # Calculate the far field effect that this multipole series has on another
    def far_field_vector(self, R_b):
        R_ab = R_b - self.R_a
        R = np.linalg.norm(R_ab)
        dmats = SphericalRotationMatrixGenerator(R_b, self.R_a)

        Vff = RealRegularHarmonics(R_b, self.lmax)

        for l in range(self.lmax+1):
            for j in range(self.lmax+1):
                T = self.build_T_spherical(l, j, R)
                rotated_mpole = np.dot(dmats.D(j), self.Rlm[j])
                nterms = 2*min(l,j)+1
                temp = np.multiply(T, rotated_mpole[:nterms])
                Vff.Rlm[l] += np.dot(dmats.D(l).T[:,:nterms], temp)

        return Vff

class RealIrregularHarmonics:

    def __init__(self, lmax, x, y, z):
        self.Ic = []
        self.Is = []
        self.I = []

        r2 = x**2 + y**2 + z**2
        r = np.sqrt(r2)

        # Recursive Building of Real Irregular Harmonics (Helgaker Equ. 9.13.85 to 9.13.89)
        for l in range(lmax+1):
            self.Ic.append(np.zeros(l+1))
            self.Is.append(np.zeros(l+1))
            self.I.append(np.zeros(2*l+1))

            if (l == 0):
                self.Ic[l][0] = 1.0 / r
                self.Is[l][0] = 0.0
            
            elif (l == 1):
                self.Ic[l][0] = z / (r2 * r)
                self.Is[l][0] = 0.0
                
                self.Ic[l][1] = -x / (r2 * r)
                self.Is[l][1] = -y / (r2 * r)
            
            else:
                # l - 1 > m terms
                for m in range(l-1):
                    self.Ic[l][m] = ((2*l-1) * z * self.Ic[l-1][m] - ((l-1)**2 - m**2) * self.Ic[l-2][m]) / r2
                    self.Is[l][m] = ((2*l-1) * z * self.Is[l-1][m] - ((l-1)**2 - m**2) * self.Is[l-2][m]) / r2

                # m = l-1 contribution
                self.Ic[l][l-1] = (2*l-1) * z * self.Ic[l-1][l-1] / r2
                self.Is[l][l-1] = (2*l-1) * z * self.Is[l-1][l-1] / r2

                # m = l contribution
                self.Ic[l][l] = -(2*l-1) * (x*self.Ic[l-1][l-1] - y*self.Is[l-1][l-1]) / r2
                self.Is[l][l] = -(2*l-1) * (y*self.Ic[l-1][l-1] + x*self.Is[l-1][l-1]) / r2

            # Helgaker Equ. 9.13.49
            # Odd indices (Sine) => m = -(ind+1)//2
            # Even indices (Cosine) => m = ind//2
            for mu in range(2*l+1):
                if (mu % 2 == 0):
                    self.I[l][mu] = self.Ic[l][mu//2]
                else:
                    # Helgaker Equ. 9.13.48
                    self.I[l][mu] = (-1)**(mu+1) * self.Is[l][(mu+1)//2]