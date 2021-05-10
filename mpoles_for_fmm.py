import decimal
import numpy as np
from math import factorial
from scipy.special import comb as cb
import time

ft = factorial

_MAX_AM = 17
_DECIMAL_PREC = 60
_saved_rsh_coefs = {}
_saved_factorials = {}

def _factorial(n):
    decimal.getcontext().prec = _DECIMAL_PREC
    if n in _saved_factorials:
        return _saved_factorials[n]

    if n == 0:
        return decimal.Decimal("1.0")
    else:
        return n * _factorial(n - 1)

class RSH_Memoize(object):
    """
    Simple memoize class for RSH_coefs which is quite expensive
    """

    def __init__(self, func):
        self.func = func
        self.mem = {}

    def __call__(self, AM, **kwargs):

        # Bypass Memoize for testing
        if kwargs.get("force_call", False):
            return self.func(AM)

        if AM not in self.mem:
            self.mem[AM] = self.func(AM)

        return self.mem[AM]


@RSH_Memoize
def cart_to_RSH_coeffs_gen(l):
    """
    Generates a coefficients [ coef, x power, y power, z power ] for each component of
    a regular solid harmonic (in terms of raw Cartesians) with angular momentum l.
    See eq. 23 of ACS, F. C. Pickard, H. F. Schaefer and B. R. Brooks, JCP, 140, 184101 (2014)
    Returns coeffs with order 0, +1, -1, +2, -2, ...
    """

    # Arbitrary precision math with 50 decimal places
    decimal.getcontext().prec = _DECIMAL_PREC

    terms = []
    for m in range(l + 1):
        thisterm = {}
        p1 = ((_factorial(l - m)) / (_factorial(l + m))).sqrt() * ((_factorial(m)) / (2**l))
        if m:
            p1 *= decimal.Decimal("2.0").sqrt()

        # Loop over cartesian components
        for lz in range(l + 1):
            for ly in range(l - lz + 1):

                lx = l - ly - lz
                xyz = lx, ly, lz
                j = int((lx + ly - m) / 2)
                if (lx + ly - m) % 2 == 1 or j < 0:
                    continue

                # P2
                p2 = decimal.Decimal(0.0)
                for i in range(int((l - m) / 2) + 1):
                    if i >= j:
                        p2 += (-1)**i * _factorial(2 * l - 2 * i) / (_factorial(l - i) * _factorial(i - j) *
                                                                     _factorial(l - m - 2 * i))

                # P3
                p3 = decimal.Decimal(0.0)
                for k in range(j + 1):
                    if (j >= k) and (lx >= 2 * k) and (m + 2 * k >= lx):
                        p3 += (-1)**k / (_factorial(j - k) * _factorial(k) * _factorial(lx - 2 * k) *
                                         _factorial(m - lx + 2 * k))

                p = p1 * p2 * p3

                # Add in part if not already present
                if xyz not in thisterm:
                    thisterm[xyz] = [decimal.Decimal(0.0), decimal.Decimal(0.0)]

                # Add the two components
                if (m - lx) % 2:
                    # imaginary
                    sign = decimal.Decimal(-1.0)**decimal.Decimal((m - lx - 1) / 2.0)
                    thisterm[xyz][1] += sign * p
                else:
                    # real
                    sign = decimal.Decimal(-1.0)**decimal.Decimal((m - lx) / 2.0)
                    thisterm[xyz][0] += sign * p

        tmp_R = []
        tmp_I = []
        for k, v in thisterm.items():
            if abs(v[0]) > 0:
                tmp_R.append((k, v[0]))
            if abs(v[1]) > 0:
                tmp_I.append((k, v[1]))

        if m == 0:
            terms.append(tmp_R)
        else:
            terms.append(tmp_R)
            terms.append(tmp_I)

    return terms

def cart_address(lx, ly, lz):
    l = lx + ly + lz
    return lz*(2*l-lz+3)//2 + ly

def binomial(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n-k)):
        b *= n
        b //= t+1
        n -= 1
    return b

def mvals(l):
    ms = [0]
    for m in range(1,l+1):
        ms.extend([m,-m])
    return ms

def build_T_spherical(la, lb, R):
    ms = mvals(min(la,lb))
    return np.array([((-1)**(lb-m)) * np.sqrt(1.0 * binomial(la+lb, la+m) * binomial(la+lb, la-m)) for m in ms]) / R**(la+lb+1)


def build_T_cartesian(la,lb, Rab, R):
    T = np.zeros(((la+1)*(la+2)//2, (lb+1)*(lb+2)//2))
    idxa = 0
    for lza in range(la+1):
        for lya in range(la-lza+1):
            lxa  = la - lya - lza
            idxb = 0
            for lzb in range(lb+1):
                for lyb in range(lb-lzb+1):
                    lxb  = lb - lyb - lzb
                    for coef,xpow,ypow,zpow,Rpow in coefficients(lxa+lxb, lya+lyb, lza+lzb):
                        T[idxa][idxb] += coef * Rab[0]**xpow * Rab[1]**ypow * Rab[2]**zpow * R**Rpow
                    idxb += 1
            idxa += 1
    return T / R**(2*la+2*lb+1)


def coefficients(lx, ly, lz):
    l = lx + ly + lz
    vals = []
    prefac = (-1)**l * factorial(lx) * factorial(ly) * factorial(lz) / 2**l
    for t in range(lx//2+1):
        xpow = lx - 2 * t
        for u in range(ly//2+1):
            ypow = ly - 2 * u
            for v in range(lz//2+1):
                zpow = lz - 2 * v
                tuv = t + u + v
                c1 = (-1)**tuv
                c2 = factorial(2*l - 2*tuv)
                c3 = factorial(l - tuv)
                c4 = factorial(t) * factorial(u) * factorial(v)
                c5 = factorial(xpow) * factorial(ypow) * factorial(zpow)
                body = c1 * c2 / ( c3 * c4 * c5)
                rpow = 2*tuv
                vals.append((prefac*body, xpow, ypow, zpow, rpow))
    return vals

class SphericalRotationMatrixGenerator:
    """ Generated rotation matrices for arbitrary order regular solid harmonics, using the
        equations from J. Phys. Chem. 1996, 100, 6342-6347, with some small typos corrected """

    def m_addr(self, m):
        """Given the type of element, return its address
           Ordering is 0, 1c, 1s, 2c, 2s..."""
        if m <= 0:
            # 0, 1s, 2s, 3s, ...
            return 2*(-m)
        else:
            # 1c, 2c, 3c, ...
            return 2*m-1

    def __init__(self, R_a, R_b):
        # Find the rotation matrix to make the internuclear vector the z axis
        Rab = R_a - R_b
        z_axis = Rab / np.linalg.norm(Rab)
        ca = z_axis.copy()
        if R_a[1] == R_b[1] and R_a[2] == R_b[2]:
            ca[1] += 1.
        else:
            ca[0] += 1.
        this_dot = ca.dot(z_axis)
        ca = ca - (z_axis * this_dot)
        x_axis = ca / np.linalg.norm(ca)
        y_axis = np.cross(z_axis, x_axis)
        self.Uz = np.array([x_axis, y_axis, z_axis])
        self.Dcache = {}

    def U(self, l, m, M):
        return self.P(0, l, m, M)

    def V(self, l, m, M):
        if m == 0:
            return self.P(1, l,  1, M) + self.P(-1, l, -1, M)
        elif m > 0:
            if m == 1:
                return np.sqrt(2)*self.P(1, l,  m-1, M)
            else:
                return self.P( 1, l,  m-1, M) - self.P(-1, l, -m+1, M)
        else:
            if m == -1:
                return np.sqrt(2)*self.P(-1, l, -m-1, M)
            else:
                return self.P( 1, l,  m+1, M) + self.P(-1, l, -m-1, M)

    def W(self, l, m, M):
        if m == 0:
            return 0.0
        elif m > 0:
            return self.P( 1, l,  m+1, M) + self.P(-1, l, -m-1, M)
        else:
            return self.P( 1, l,  m-1, M) - self.P(-1, l, -m+1, M)

    def u(self, l, m, M):
        if abs(M) < l:
            return np.sqrt((l+m)*(l-m)/((l+M)*(l-M)))
        elif abs(M) == l:
            return np.sqrt((l+m)*(l-m)/((2*l)*(2*l-1)))

    def v(self, l, m, M):
        dm0 = 1 if m==0 else 0
        if abs(M) < l:
            return (1-2*dm0) * np.sqrt((1+dm0)*(l+abs(m)-1)*(l+abs(m)) / ((l+M)*(l-M))) / 2
        elif abs(M) == l:
            return (1-2*dm0) * np.sqrt((1+dm0)*(l+abs(m)-1)*(l+abs(m)) / ((2*l)*(2*l-1))) / 2

    def w(self, l, m, M):
        dm0 = 1 if m==0 else 0
        if abs(M) < l:
            return (dm0-1) * np.sqrt((l-abs(m)-1)*(l-abs(m)) / ((l+M)*(l-M))) / 2
        elif abs(M) == l:
            return (dm0-1) * np.sqrt((l-abs(m)-1)*(l-abs(m)) / ((2*l)*(2*l-1))) / 2

    def P(self, i, l, mu, M):
        I = self.m_addr(i)
        D1 = self.D(1)
        Dl1 = self.D(l-1)
        if abs(M) < l:
            return D1[I, self.m_addr( 0)] * Dl1[self.m_addr(mu), self.m_addr(M)]
        elif M == l:
            return D1[I, self.m_addr( 1)] * Dl1[self.m_addr(mu), self.m_addr( M-1)] - D1[I, self.m_addr(-1)] * Dl1[self.m_addr(mu), self.m_addr(-M+1)]
        elif M == -l:
            return D1[I, self.m_addr( 1)] * Dl1[self.m_addr(mu), self.m_addr( M+1)] + D1[I, self.m_addr(-1)] * Dl1[self.m_addr(mu), self.m_addr(-M-1)]

    def D(self, l):
        # Memoize the return value
        if l in self.Dcache:
            return self.Dcache[l]

        if l == 0:
            self.Dcache[l] = np.array([[1.0]])
        elif l==1:
            # Permute to spherical harmonic order to get the dipole rotation matrix
            perms = [2, 0, 1]
            self.Dcache[l] = np.take(np.take(self.Uz, perms, axis=0), perms, axis=1)
        else:
            # Build D by recursion
            dim = 2*l + 1
            Dmat = np.zeros((dim, dim))
            for m in mvals(l):
                k1 = self.m_addr(m)
                for M in mvals(l):
                    k2 = self.m_addr(M)
                    Uterm = self.u(l, m, M)
                    if Uterm:
                        Uterm *= self.U(l, m, M)
                    Vterm = self.v(l, m, M)
                    if Vterm:
                        Vterm *= self.V(l, m, M)
                    Wterm = self.w(l, m, M)
                    if Wterm:
                        Wterm *= self.W(l, m, M)
                    Dmat[k1,k2] = (Uterm + Vterm + Wterm)
            self.Dcache[l] = Dmat
        return self.Dcache[l]

def compute_Vff_sphe(R_a, R_b, mpoles_sphe, lmax):
    R_ab = R_b - R_a
    R = np.linalg.norm(R_ab)
    dmats = SphericalRotationMatrixGenerator(R_a, R_b)
    Vff = []
    for la in range(lmax+1):
        Vff.append([])
        rotated_a = np.array(np.dot(dmats.D(la), mpoles_sphe[la]))
        for lb in range(lmax+1):
            Vff[la].append([])
            l = min(la,lb)
            nterms = 2*l+1
            V = np.einsum('a,a->a', rotated_a[:nterms], build_T_spherical(la, lb, R))
            V = np.array(np.dot(dmats.D(l).T, V))
            Vff[la][lb] = V
    return Vff

def energies_Vff_sphe(R_a, R_b, mpoles_sphe, Vff, lmax):
    E = 0.0
    dmats = SphericalRotationMatrixGenerator(R_a, R_b)
    for la in range(lmax+1):
        for lb in range(lmax+1):
            l = min(la, lb)
            nterms = 2*l+1
            V = np.dot(dmats.D(l), Vff[la][lb])
            rot_mpoles = np.array(np.dot(dmats.D(lb), mpoles_sphe[lb]))
            E += np.einsum('a,a', rot_mpoles[:nterms], V)
    return E

def compute_terms_sphe(R_a, R_b, mpoles_sphe_a, mpoles_sphe_b, lmax):
    R_ab = R_b - R_a
    R = np.linalg.norm(R_ab)
    dmats = SphericalRotationMatrixGenerator(R_a, R_b)
    Evals = np.zeros((lmax+1, lmax+1))
    for la in range(lmax+1):
        for lb in range(lmax+1):
            rotated_a = np.array(np.dot(dmats.D(la), mpoles_sphe_a[la]))
            rotated_b = np.array(np.dot(dmats.D(lb), mpoles_sphe_b[lb]))
            nterms = 2*min(la,lb)+1
            E = np.einsum('a,a,a', rotated_a[:nterms], build_T_spherical(la,lb,R), rotated_b[:nterms])
            Evals[la,lb] = E
    return Evals

# Uses rotation matrices to translate multipoles from R_a to R_b
def translate_mpoles_sphe(R_a, R_b, mpoles_sphe, lmax):
    rot_mpoles = []
    new_mpoles = []
    fin_mpoles = []
    R_ab = R_b - R_a
    R = np.linalg.norm(R_ab)
    dmats = SphericalRotationMatrixGenerator(R_a, R_b)

    for la in range(lmax+1):
        rot_mpoles.append(np.zeros(2*la+1, dtype=np.double))
        new_mpoles.append(np.zeros(2*la+1, dtype=np.double))
        fin_mpoles.append(np.zeros(2*la+1, dtype=np.double))

    # Rotate Multipoles to direction of translation
    for l in range(0, lmax+1):
        rot_mpoles[l] = np.array(np.dot(dmats.D(l), mpoles_sphe[l]))

    # Translate rotated multipoles
    for l in range(0, lmax+1):
        for j in range(0, l+1):
            for m in range(-j, j+1):

                if m <= 0:
                    mp = 2*(-m)
                else:
                    mp = 2*m-1

                prefactor = np.sqrt(cb(l+m,j+m)*cb(l-m,j-m))
                new_mpoles[l][mp] += prefactor * (-R)**(l-j) * rot_mpoles[j][mp]

    # Back-rotate multipole back to original basis
    for l in range(0, lmax+1):
        fin_mpoles[l] = np.array(np.dot(dmats.D(l).T, new_mpoles[l]))

    return fin_mpoles
            

def compute_terms_cart(R_a, R_b, mpoles_cart):
    R_ab = R_b - R_a
    R = np.linalg.norm(R_ab)
    Evals = np.zeros((lmax+1, lmax+1))
    for la in range(lmax+1):
        for lb in range(lmax+1):
            E = (-1)**lb * np.einsum('a,ab,b', mpoles_cart[la], build_T_cartesian(la,lb,R_ab,R), mpoles_cart[lb])
            Evals[la,lb] = E
    return Evals

def compute_mpoles_spherical(l, dx, dy, dz):
    sphe = np.zeros(2*l + 1)
    for n, contributions in enumerate(cart_to_RSH_coeffs_gen(l)):
        val = 0.0
        for lvals, coef in contributions:
            lx, ly, lz = lvals
            # Account for permutations.  I'm not sure why, but this is needed
            # prefac = factorial(lx)*factorial(ly)*factorial(lz)
            val += float(coef) * (dx**lx) * (dy**ly) * (dz**lz)
        sphe[n] = val   
    return sphe

# Calculates elements of X transformation matrix
def X_mat(m1, m2):
    if (m1 == 0 and m2 == 0):
        return 1.0
    if (m1 == m2 and m1 > 0):
        return (-1)**m1 / np.sqrt(2)
    if (m1 == -m2 and m1 > 0):
        return 1.0 / np.sqrt(2)
    if (m1 == -m2 and m1 < 0):
        return (-1)**m1 * (1.0j) / np.sqrt(2)
    if (m1 == m2 and m1 < 0):
        return (-1.0j) / np.sqrt(2)
    else:
        return 0.0

test = """
# Set positions
ra = np.array([ 0.3, 2.8, 10.0 ])
rb = np.array([ 4.0, 3.0, -2.0 ])

# Make some random multipoles
lmax = 10
mpoles_cart = []
mpoles_sphe = []
for l in range(lmax+1):
    cart = 2*np.random.rand(((l+1)*(l+2)//2))-1

    sphe = []
    for contributions in cart_to_RSH_coeffs_gen(l):
        val = 0.0
        for lvals,coef in contributions:
            lx,ly,lz = lvals
            # Account for permutations.  I'm not sure why, but this is needed
            prefac = factorial(lx)*factorial(ly)*factorial(lz)
            val += prefac * float(coef) * cart[cart_address(*lvals)]
        sphe.append(val)
    mpoles_cart.append(np.array(cart))
    mpoles_sphe.append(np.array(sphe))

cart1 = time.perf_counter()
Ecart = compute_terms_cart(ra, rb, mpoles_cart)
cart2 = time.perf_counter()

sphe1 = time.perf_counter()
Esphe = compute_terms_sphe(ra, rb, mpoles_sphe, mpoles_sphe, lmax)
sphe2 = time.perf_counter()

print(f"Time taken for Cartesian = {cart2-cart1:.3f} seconds")
print(f"Time taken for Spherical = {sphe2-sphe1:.3f} seconds")
print("Errors for each angular momentum pair")
np.set_printoptions(suppress=True)
print(Ecart-Esphe)
"""