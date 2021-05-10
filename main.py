from fmm import *
import numpy as np
import scipy as sp
from math import factorial
from mpoles_for_fmm import *
import time

def run_test1():
    q = 100
    pc1 = PointCharge(q, 0.0, 0.0, 0.0)
    pc2 = PointCharge(q, 2.0, 0.0, 0.0)
    pc3 = PointCharge(q, 0.0, 2.0, 0.0)
    pc4 = PointCharge(q, 2.0, 2.0, 0.0)
    pc5 = PointCharge(q, 2.0, 0.0, 2.0)
    pc6 = PointCharge(q, 0.0, 2.0, 2.0)
    pc7 = PointCharge(q, 0.0, 0.0, 2.0)
    pc8 = PointCharge(-q, 2.0, 2.0, 2.0)
    pc9 = PointCharge(q, 1.6, 1.6, 1.6)
    pc10 = PointCharge(q, 0.4, 0.4, 0.4)

    charges = [pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8, pc9, pc10]

    pairwise = PairwisePointChargeSystem(charges)
    fmmTree1 = FMMTree(1, charges, 4, ws=2)
    fmmTree2 = FMMTree(4, charges, 4, ws=2)

    print(pairwise.get_energy())
    print(fmmTree1.get_energy())
    print(fmmTree2.get_energy())

def run_test2():
    charges = []

    for i in range(100):
        x = 0.1 * np.random.rand()
        y = 0.1 * np.random.rand()
        z = 0.1 * np.random.rand()
        q = np.random.randint(-11, 10)

        charges.append(PointCharge(q, x + 25.0, y + 25.0, z + 25.0))
    
    for i in range(100):
        x = 0.1 * np.random.rand()
        y = 0.1 * np.random.rand()
        z = 0.1 * np.random.rand()
        q = np.random.randint(-11, 10)

        charges.append(PointCharge(q, x + 75.0, y + 75.0, z + 75.0))
    
    pairwise = PairwisePointChargeSystem(charges)
    
    fmmTree2 = FMMTree(2, charges, 17)

    child1 = fmmTree2.root.children['000']
    child2 = fmmTree2.root.children['111']

    c1r = np.array([child1.xM, child1.yM, child1.zM])
    c2r = np.array([child2.xM, child2.yM, child2.zM])

    c1pw = PairwisePointChargeSystem(child1.charges)
    c2pw = PairwisePointChargeSystem(child2.charges)

    Evals = compute_terms_sphe(c1r, c2r, child1.multipoles, child2.multipoles, 17)
    FF_E = np.einsum('ab->', Evals)

    print(pairwise.get_energy())
    print(FF_E)
    print(c1pw.get_energy() + c2pw.get_energy() + FF_E)

    Vff = compute_Vff_sphe(c1r, c2r, child2.multipoles, 17)
    FF_E2 = energies_Vff_sphe(c1r, c2r, child1.multipoles, Vff, 17)
    print(FF_E2)

def run_test3():
    charges = []

    np.random.seed(65536)

    for i in range(500):
        q = np.random.randint(-20, 21)
        x = 500.0 + 500.0 * np.random.rand()
        y = 500.0 + 500.0 * np.random.rand()
        z = 500.0 + 500.0 * np.random.rand()
        charges.append(PointCharge(q, x, y, z))

    start = time.time()
    pairwise = PairwisePointChargeSystem(charges)
    correct = pairwise.get_energy()
    print(correct)
    print(time.time() - start)

    REAL = 34.02277360858226

    fmmTree4 = FMMTree(3, charges, 6, ws=2)
    test = fmmTree4.get_energy()
    print(test)
    print(fmmTree4.timings)

    print(abs(test - correct))

def run_test4():
    mp1 = np.array([4.0])
    dp1 = np.array([3.0, -2.0, -9.0])
    qp1 = np.array([9.0, -7.0, 5.2, -9.8, 0.5])

    mp2 = np.array([-3.0])
    dp2 = np.array([1.0, 7.0, 4.0])
    qp2 = np.array([9.8, 9.1, -5.4, -4.1, 3.2])

    R1 = np.array([0.0, 0.0, 0.0])
    Rh = np.array([3.5, 2.1, 0.9])
    R2 = np.array([7.1, 4.2, 1.9])

    rrh1 = RealRegularHarmonics(R1, 2)
    rrh2 = RealRegularHarmonics(R2, 2)

    rrh1.Rlm[0] = mp1
    rrh1.Rlm[1] = dp1
    rrh1.Rlm[2] = qp1

    rrh2.Rlm[0] = mp2
    rrh2.Rlm[1] = dp2
    rrh2.Rlm[2] = qp2

    Vff1 = rrh1.far_field_vector(R2)
    Vff2 = rrh2.far_field_vector(R1)

    # Right answer
    ref = compute_terms_sphe(R1, R2, [mp1, dp1, qp1], [mp2, dp2, qp2], lmax=2)
    print(np.einsum('ab->', ref))

    # Test answer 1
    test = 0.0
    for l in range(3):
        test += np.dot(Vff2.Rlm[l], rrh1.Rlm[l])
    print(test)

    # Test answer 2
    test2 = 0.0
    for l in range(3):
        test2 += np.dot(Vff1.Rlm[l], rrh2.Rlm[l])
    print(test2)

def run_test5():

    np.random.seed(42)
    mp1 = []
    mp2 = []
    for l in range(6):
        mp1.append(10.0 * np.random.rand(2*l+1))
        mp2.append(10.0 * np.random.rand(2*l+1))

    R1 = np.array([0.0, 0.0, 0.0])
    Rh = np.array([0.0, 0.0, 99.0])
    R2 = np.array([0.0, 0.0, 100.0])

    rrh1 = RealRegularHarmonics(R1, 5)
    rrh2 = RealRegularHarmonics(R2, 5)

    rrh1.Rlm = mp1
    rrh2.Rlm = mp2

    # Right answer
    ref = compute_terms_sphe(R1, R2, mp1, mp2, lmax=5)
    print(np.einsum('ab->', ref))

    Vff1 = rrh1.far_field_vector(Rh)
    Vff1 = Vff1.irregular_translate(R2)

    # Test answer
    test = 0.0
    for l in range(6):
        test += np.dot(Vff1.Rlm[l], rrh2.Rlm[l])
    print(test)

def test6():
    mp1 = np.array([27.0])
    mp2 = np.array([-33.0])

    # dp1 = np.array([4.0, -2.0, 5.0])
    # dp2 = np.array([-3.0, 3.0, 5.0])
    dp1 = np.zeros(3)
    dp2 = np.zeros(3)

    R1 = np.array([0.0, 0.0, 0.0])
    Rh = np.array([0.0, 0.0, 99.0])
    R2 = np.array([0.0, 0.0, 100.0])

    rrh1 = RealRegularHarmonics(R1, 1)
    rrh2 = RealRegularHarmonics(R2, 1)

    rrh1.Rlm[0] = mp1
    rrh1.Rlm[1] = dp1

    rrh2.Rlm[0] = mp2
    rrh2.Rlm[1] = dp2

    # Right answer
    ref = compute_terms_sphe(R1, R2, [mp1, dp1], [mp2, dp2], lmax=1)
    print(np.einsum('ab->', ref))

    Vff1 = rrh1.far_field_vector(Rh)
    Vff1 = Vff1.irregular_translate(R2)

    # Test answer
    test = 0.0
    for l in range(2):
        test += np.dot(Vff1.Rlm[l], rrh2.Rlm[l])
    print(test)


if __name__ == '__main__':
    run_test3()