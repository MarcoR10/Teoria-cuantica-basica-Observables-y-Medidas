# Marco Alvarez
# Teoría cuántica básica, Observables y Medidas

import numpy as np
import NComplex as nc
import VComplex as vc


#---------------- Retos De Programacion ---------------------------#

def hermitiana(a):
    if a == vc.matrixAdj(a):
        return True
    else:
        return False


def valorEsperado(observable, estado):
    aux = vc.Accion(observable, estado)
    ve = vc.transito(aux, estado)
    return ve


def varianza(matriz, ket):
    ve = valorEsperado(matriz, ket)
    m, n = len(matriz), len(matriz[0])
    b = vc.iden(m, n)
    d = vc.multiMatrix(b, nc.producplx(ve, (-1, 0)))
    delta = vc.sumatriz(matriz, d)
    preva = vc.multiMatrix(delta, delta)
    var = valorEsperado(preva, ket)
    return var


def proba(a, b):
    aux = vc.ProductIntVec(b, a)
    res = (nc.modulo(aux)) ** 2
    return res


# --------------------------- Ejercicios ----------------------------------- #


# 4.3.1
def posi(a, indice):
    s = [[(0, 1), (1, 0)], [(0, -1), (1, 0)], [(1, 0), (1, 0)], [(-1, 0), (1, 0)], [(0, 0), (1, 0)], [(1, 0), (0, 0)]]
    result = []
    for i in range((indice * 2) - 2, indice * 2):
        if proba(a, s[i]) != 0.0:
            result = result + s[i]
    return result


# 4.3.2
def cal(a, indice):
    matrices = [[[(1, 0), (0, 0)], [(0, 0), (-1, 0)]], [[(0, 0), (0, -1)], [(0, 1), (0, 0)]],
                [[(0, 0), (1, 0)], [(1, 0), (0, 0)]]]
    vp = []
    aux = posi(a, indice)
    probas = []
    res = 0
    for i in range(3):
        valores, no = np.linalg.eig(matrices[i])
        vp = vp + valores
    for i in range(len(aux)):
        probas = probas + proba(a, aux[i])
    for i in range(2):
        res = res + (proba[i] * vp[indice][i])
    return res


# 4.4.1
def comp(U1, U2):
    if vc.uni(U1) and vc.uni(U2):
        aux = vc.multiMatrix(U1, U2)
        return vc.uni(aux)


# 4.4.2
def billar(matriz, vector):
    for i in range(3):
        vector = vc.Accion(matriz, vector)
    proba = (nc.modulo(vector[3])) ** 2
    return proba

# -------------------------------------------------------------- #
