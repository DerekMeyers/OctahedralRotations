from __future__ import division
import sys
import numpy as np
from collections import OrderedDict
from functools import reduce
from lmfit import Model, conf_interval, ci_report

""" Using pseudocubic notation for H,K,L and a,b,c everywhere"""

import periodictable as pt
import periodictable.cromermann as ptc
# https://pypi.python.org/pypi/periodictable
# download and python setup.py install from folder

def get_form_factor(element_name, wavelength):
    f_obj = getattr(pt, element_name)
    tup_re_im = f_obj.xray.scattering_factors(wavelength=wavelength)
    return np.array([tup_re_im[0] + 1j*tup_re_im[1]])

def CM(H, K, L, a, b,  c, symbol):
    return np.array(ptc.fxrayatstol(symbol, 1/(2*dHKL(H,K,L,a,b,c))))

def get_eta(omoffset, theta, chi):
    return np.arcsin(np.sin(np.radians(omoffset+theta))*np.sin(np.radians(chi)))

def tand(theta):
    """ tan in degrees """
    return np.tan(np.radians(theta))

def sind(theta):
    """ sin in degrees """
    return np.sin(np.radians(theta))

def arcsind(theta):
    """ arcsin gives degrees """
    return 180/np.pi*np.arcsin(theta)

def calc_eta(theta,chi):
    return np.abs(180/np.pi*np.arcsin(np.sin(np.radians(theta))*np.sin(np.radians(chi))))
    
def get_cell_1RSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_1RSL = OrderedDict(
    O1 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O2 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O3 =  [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O4 =  [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O5 =  [0,0.25 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O6 =  [0.25 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O7 =  [0.5, 0.25 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O8 =  [0.25 + b/(4*a)*tand(gamma1), 0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O9 =  [0, 0.75 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O10 = [0.5,0.75 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O11 = [0.75 - b/(4*a)*tand(gamma1), 0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O12 = [0.75 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O13 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O14 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O15 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O16 = [0.75 + c/(4*a)*tand(beta2), 0.25 - c/(4*b) * tand(alpha2),0.5],
    O17 = [0, 0.25 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O18 = [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O19 = [0.5,0.25 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O20 = [0.25 + b/(4*a)*tand(gamma2), 0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O21 = [0, 0.75 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5,0.75 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - b/(4*a)*tand(gamma2), 0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O24 = [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],

    A1 = [d1, d1 + d2, 0],
    A2 = [0.5+d1+d2, d1, 0],
    A3 = [0.5+d1, 0.5+d1+d2, 0],
    A4 = [d1+d2, 0.5+d1, 0],
    A5 = [-d1, -d1-d2, 0.5],
    A6 = [0.5-d1-d2, -d1, 0.5],
    A7 = [0.5-d1, 0.5-d1-d2, 0.5],
    A8 = [-d1-d2, 0.5-d1, 0.5],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    B3 = [0.75, 0.75, 0.25],
    B4 = [0.25, 0.75, 0.25],
    
    C1 = [0.25, 0.25, 0.75],
    C2 = [0.75, 0.25, 0.75],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_1RSL

def get_cell_2RSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_2RSL = OrderedDict(
    O1 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O2 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O3 =  [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O4 =  [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O5 =  [0,0.25 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O6 =  [0.25 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O7 =  [0.5, 0.25 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O8 =  [0.25 + b/(4*a)*tand(gamma1), 0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O9 =  [0, 0.75 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O10 = [0.5,0.75 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O11 = [0.75 - b/(4*a)*tand(gamma1), 0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O12 = [0.75 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O13 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O14 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O15 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O16 = [0.75 + c/(4*a)*tand(beta2), 0.25 - c/(4*b) * tand(alpha2),0.5],
    O17 = [0, 0.25 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O18 = [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O19 = [0.5,0.25 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O20 = [0.25 + b/(4*a)*tand(gamma2), 0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O21 = [0, 0.75 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5,0.75 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - b/(4*a)*tand(gamma2), 0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O24 = [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
        
    A1 = [-d1, d1 + d2, 0],
    A2 = [0.5-d1-d2, d1, 0],
    A3 = [0.5-d1, 0.5+d1+d2, 0],
    A4 = [-d1-d2, 0.5+d1, 0],
    A5 = [d1, -d1-d2, 0.5],
    A6 = [0.5+d1+d2, -d1, 0.5],
    A7 = [0.5+d1, 0.5-d1-d2, 0.5],
    A8 = [d1+d2, 0.5-d1, 0.5],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    B3 = [0.75, 0.75, 0.25],
    B4 = [0.25, 0.75, 0.25],

    C1 = [0.25, 0.25, 0.75],
    C2 = [0.75, 0.25, 0.75],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_2RSL

def get_cell_1LSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_1LSL = OrderedDict(
    O1 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O2 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O3 =  [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O4 =  [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O5 =  [0,0.25 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O6 =  [0.25 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O7 =  [0.5, 0.25 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O8 =  [0.25 + b/(4*a)*tand(gamma1), 0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O9 =  [0, 0.75 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O10 = [0.5,0.75 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O11 = [0.75 - b/(4*a)*tand(gamma1), 0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O12 = [0.75 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O13 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O14 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O15 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O16 = [0.75 + c/(4*a)*tand(beta2), 0.25 - c/(4*b) * tand(alpha2),0.5],
    O17 = [0, 0.25 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O18 = [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O19 = [0.5,0.25 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O20 = [0.25 + b/(4*a)*tand(gamma2), 0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O21 = [0, 0.75 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5,0.75 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - b/(4*a)*tand(gamma2), 0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O24 = [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
        
    A1 = [d1+d2, d1, 0],
    A2 = [0.5+d1, d1+d2, 0],
    A3 = [0.5+d1+d2, 0.5+d1, 0],
    A4 = [d1, 0.5+d1+d2, 0],
    A5 = [-d1-d2, -d1, 0.5],
    A6 = [0.5-d1, -d1-d2, 0.5],
    A7 = [0.5-d1-d2, 0.5-d1, 0.5],
    A8 = [-d1, 0.5-d1-d2, 0.5],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    B3 = [0.75, 0.75, 0.25],
    B4 = [0.25, 0.75, 0.25],

    C1 = [0.25, 0.25, 0.75],
    C2 = [0.75, 0.25, 0.75],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_1LSL

def get_cell_2LSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_2LSL = OrderedDict(
    O1 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O2 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O3 =  [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O4 =  [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.],
    O5 =  [0,0.25 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O6 =  [0.25 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O7 =  [0.5, 0.25 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O8 =  [0.25 + b/(4*a)*tand(gamma1), 0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O9 =  [0, 0.75 - a/(4*b)*tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O10 = [0.5,0.75 + a/(4*b)*tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O11 = [0.75 - b/(4*a)*tand(gamma1), 0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O12 = [0.75 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O13 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O14 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O15 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2,0.5],
    O16 = [0.75 + c/(4*a)*tand(beta2), 0.25 - c/(4*b) * tand(alpha2),0.5],
    O17 = [0, 0.25 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O18 = [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O19 = [0.5,0.25 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O20 = [0.25 + b/(4*a)*tand(gamma2), 0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O21 = [0, 0.75 - a/(4*b)*tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5,0.75 + a/(4*b)*tand(gamma2), 0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - b/(4*a)*tand(gamma2), 0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O24 = [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],


    A1 = [-d1-d2, d1, 0],
    A2 = [0.5-d1, d1+d2, 0],
    A3 = [0.5-d1-d2, 0.5+d1, 0],
    A4 = [-d1, 0.5+d1+d2, 0],
    A5 = [d1+d2, -d1, 0.5],
    A6 = [0.5+d1, -d1-d2, 0.5],
    A7 = [0.5+d1+d2, 0.5-d1, 0.5],
    A8 = [d1, 0.5-d1-d2, 0.5],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    B3 = [0.75, 0.75, 0.25],
    B4 = [0.25, 0.75, 0.25],
 
    C1 = [0.25, 0.25, 0.75],
    C2 = [0.75, 0.25, 0.75],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_2LSL

# Below are the cells for in-plane in-phase tilts. It is not a simply transformation as before as the  different alpha, beta,
# and gamma angles  are always along c-direction. Also note the B and C-sites do not change  as they would with a transform.

def get_cell_1RSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_1RSL_IP = OrderedDict(
    O1 =  [0.25 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O2 =  [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
    O3 =  [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O4 =  [0.75 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O5 =  [0, 0.25 - a/(4*b) * tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O6 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O7 =  [0.5, 0.25 + a/(4*b) * tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O8 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O9 =  [0,  0.25 + a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O10 = [0.5, 0.25 - a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O11 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O12 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2,  0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O13 = [0.25 - b/(4*a)*tand(gamma1),0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O14 = [0.25 + b/(4*a)*tand(gamma2),0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O15 = [0.75 - b/(4*a)*tand(gamma2),0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O16 = [0.75 + b/(4*a)*tand(gamma1),0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O17 = [0, 0.75 + a/(4*b) * tand(gamma1), 0.25 + a/(4*c)* tand(beta1)],
    O18 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O19 = [0.5, 0.75 - a/(4*b) * tand(gamma1),0.25 - a/(4*c)* tand(beta1)],
    O20 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O21 = [0,  0.75 - a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5, 0.75 + a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O24 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],

    A1 = [d1, 0, d1 + d2],
    A2 = [0.5+d1+d2, 0, d1],
    A3 = [0.5+d1, 0, 0.5+d1+d2],
    A4 = [d1+d2, 0, 0.5+d1],
    A5 = [-d1, 0.5, -d1-d2],
    A6 = [0.5-d1-d2, 0.5, -d1],
    A7 = [0.5-d1, 0.5, 0.5-d1-d2],
    A8 = [-d1-d2, 0.5, 0.5-d1],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    C1 = [0.75, 0.25, 0.75],
    C2 = [0.25, 0.25, 0.75],
 
    B3 = [0.25, 0.75, 0.25],
    B4 = [0.75, 0.75, 0.25],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_1RSL_IP


def get_cell_2RSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_2RSL_IP = OrderedDict(
    O1 =  [0.25 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O2 =  [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
    O3 =  [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O4 =  [0.75 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O5 =  [0, 0.25 - a/(4*b) * tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O6 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O7 =  [0.5, 0.25 + a/(4*b) * tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O8 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O9 =  [0,  0.25 + a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O10 = [0.5, 0.25 - a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O11 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O12 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2,  0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O13 = [0.25 - b/(4*a)*tand(gamma1),0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O14 = [0.25 + b/(4*a)*tand(gamma2),0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O15 = [0.75 - b/(4*a)*tand(gamma2),0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O16 = [0.75 + b/(4*a)*tand(gamma1),0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O17 = [0, 0.75 + a/(4*b) * tand(gamma1), 0.25 + a/(4*c)* tand(beta1)],
    O18 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O19 = [0.5, 0.75 - a/(4*b) * tand(gamma1),0.25 - a/(4*c)* tand(beta1)],
    O20 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O21 = [0,  0.75 - a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5, 0.75 + a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O24 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],

    A1 = [-d1, 0, d1 + d2],
    A2 = [0.5-d1-d2, 0, d1],
    A3 = [0.5-d1, 0, 0.5+d1+d2],
    A4 = [-d1-d2, 0, 0.5+d1],
    A5 = [d1, 0.5, -d1-d2],
    A6 = [0.5+d1+d2,0.5,  -d1],
    A7 = [0.5+d1,0.5,  0.5-d1-d2],
    A8 = [d1+d2,0.5,  0.5-d1],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    C1 = [0.75, 0.25, 0.75],
    C2 = [0.25, 0.25, 0.75],
 
    B3 = [0.25, 0.75, 0.25],
    B4 = [0.75, 0.75, 0.25],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_2RSL_IP

def get_cell_1LSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_1LSL_IP = OrderedDict(
    O1 =  [0.25 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O2 =  [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
    O3 =  [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O4 =  [0.75 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O5 =  [0, 0.25 - a/(4*b) * tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O6 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O7 =  [0.5, 0.25 + a/(4*b) * tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O8 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O9 =  [0,  0.25 + a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O10 = [0.5, 0.25 - a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O11 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O12 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2,  0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O13 = [0.25 - b/(4*a)*tand(gamma1),0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O14 = [0.25 + b/(4*a)*tand(gamma2),0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O15 = [0.75 - b/(4*a)*tand(gamma2),0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O16 = [0.75 + b/(4*a)*tand(gamma1),0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O17 = [0, 0.75 + a/(4*b) * tand(gamma1), 0.25 + a/(4*c)* tand(beta1)],
    O18 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O19 = [0.5, 0.75 - a/(4*b) * tand(gamma1),0.25 - a/(4*c)* tand(beta1)],
    O20 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O21 = [0,  0.75 - a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5, 0.75 + a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O24 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],

    A1 = [d1+d2,0, d1],
    A2 = [0.5+d1,0, d1+d2],
    A3 = [0.5+d1+d2,0, 0.5+d1],
    A4 = [d1, 0, 0.5+d1+d2],
    A5 = [-d1-d2, 0.5,  -d1],
    A6 = [0.5-d1, 0.5, -d1-d2],
    A7 = [0.5-d1-d2, 0.5, 0.5-d1],
    A8 = [-d1, 0.5, 0.5-d1-d2],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    C1 = [0.75, 0.25, 0.75],
    C2 = [0.25, 0.25, 0.75],
 
    B3 = [0.25, 0.75, 0.25],
    B4 = [0.75, 0.75, 0.25],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_1LSL_IP

def get_cell_2LSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2):
    cell_2LSL_IP = OrderedDict(
    O1 =  [0.25 + b/(4*a)*tand(gamma1), 0, 0.25 - b/(4*c) * tand(alpha1)],
    O2 =  [0.25 - b/(4*a)*tand(gamma2), 0, 0.75 + b/(4*c) * tand(alpha2)],
    O3 =  [0.75 + b/(4*a)*tand(gamma2), 0, 0.75 - b/(4*c) * tand(alpha2)],
    O4 =  [0.75 - b/(4*a)*tand(gamma1), 0, 0.25 + b/(4*c) * tand(alpha1)],
    O5 =  [0, 0.25 - a/(4*b) * tand(gamma1), 0.25 + a/(4*c) * tand(beta1)],
    O6 =  [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O7 =  [0.5, 0.25 + a/(4*b) * tand(gamma1), 0.25 - a/(4*c) * tand(beta1)],
    O8 =  [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O9 =  [0,  0.25 + a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O10 = [0.5, 0.25 - a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O11 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.25 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O12 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2,  0.25 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O13 = [0.25 - b/(4*a)*tand(gamma1),0.5, 0.25 + b/(4*c) * tand(alpha1)],
    O14 = [0.25 + b/(4*a)*tand(gamma2),0.5, 0.75 - b/(4*c) * tand(alpha2)],
    O15 = [0.75 - b/(4*a)*tand(gamma2),0.5, 0.75 + b/(4*c) * tand(alpha2)],
    O16 = [0.75 + b/(4*a)*tand(gamma1),0.5, 0.25 - b/(4*c) * tand(alpha1)],
    O17 = [0, 0.75 + a/(4*b) * tand(gamma1), 0.25 + a/(4*c)* tand(beta1)],
    O18 = [0.25 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],
    O19 = [0.5, 0.75 - a/(4*b) * tand(gamma1),0.25 - a/(4*c)* tand(beta1)],
    O20 = [0.25 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O21 = [0,  0.75 - a/(4*b) * tand(gamma2), 0.75 - a/(4*c) * tand(beta2)],
    O22 = [0.5, 0.75 + a/(4*b) * tand(gamma2),0.75 + a/(4*c) * tand(beta2)],
    O23 = [0.75 - c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 - c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0.5],
    O24 = [0.75 + c/(4*a)*(tand(beta1)+tand(beta2))/2, 0.75 + c/(4*b) * (tand(alpha1)+tand(alpha2))/2, 0],

    A1 = [-d1-d2, 0, d1],
    A2 = [0.5-d1, 0, d1+d2],
    A3 = [0.5-d1-d2, 0, 0.5+d1],
    A4 = [-d1, 0, 0.5+d1+d2],
    A5 = [d1+d2, 0.5,  -d1],
    A6 = [0.5+d1, 0.5,  -d1-d2],
    A7 = [0.5+d1+d2,0.5,  0.5-d1],
    A8 = [d1, 0.5, 0.5-d1-d2],

    B1 = [0.25, 0.25, 0.25],
    B2 = [0.75, 0.25, 0.25],
    C1 = [0.75, 0.25, 0.75],
    C2 = [0.25, 0.25, 0.75],
 
    B3 = [0.25, 0.75, 0.25],
    B4 = [0.75, 0.75, 0.25],
    C3 = [0.75, 0.75, 0.75],
    C4 = [0.25, 0.75, 0.75]
    )

    return cell_2LSL_IP


def calc_FSL(H, K, L, cell, a,b,c, symbols):
    """  For each element add the contributions to the structure factor for each site. Then total structure factor is the contribution times the elemental form factor. The 2* is for the doubled unit cell as May does in paper.
     """
    def expQr(H, K, L, r):
        return np.exp(2.*np.pi*1j* (H*r[0] + K*r[1] + L*r[2]) )
    A_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] is 'A'))
    B_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] is 'B'))
    C_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] is 'C'))
    O_term = reduce(lambda x, y: x+y, (expQr(2.*H, 2.*K, 2.*L, r) for key, r in cell.items() if key[0] is 'O'))

    F_HKL = CM(H,K,L,a,b,c,symbols[0]) * A_term + CM(H, K, L, a,b, c, symbols[1]) * B_term + CM(H, K, L, a,b, c, symbols[2]) * C_term + CM(H, K, L, a,b, c, symbols[3]) * O_term
    return F_HKL


def dHKL(H,K,L,a,b,c):
    """ Calculates dHKL for orthorhombic structure with the HKL and a,c values.
    """
    dHKL = np.sqrt(1./((H/a)**2.+(K/a)**2.+(L/c)**2.))
    return dHKL

def LorenP(H, K, L, a,b, c, wavelength):
    """  LP = 1/sin(2theta).  Theta is found from Bragg's law using dHKL, for orthorhombic. Can generalize later
    for monoclinic, but change will be extremely small considering typical deviation of angle is ~ 0.1 degrees.
    """
    LP =  1./np.sin(2.*np.arcsin(wavelength/(2.*dHKL(H,K,L,a,b,c))))
    return LP

def intensity_ambmcpSL(H, K, L, eta, alpha1, beta1, gamma1, alpha2, beta2, gamma2, d1, d2, D_1Rppp, D_2Rpmp, D_1Lppm, D_2Rmpp,   D_2Rppp, D_1Rpmp, D_2Lppm, D_1Rmpp, D_2Lpmp, D_1Rppm, D_1Lpmp, D_2Rppm, a, b, c, symbols, wavelength, I0,Rot_type):
    """ Takes all positive alpha beta gamma. In-plane"""
    if Rot_type == 'In-plane':
        # The in-plane cells have y / z coordinates, c / b, and beta / gamma swapped.  For SL the B-sites have to remain ordered along c-direction, and the  seperate rotation angles must remain along c as well. This makes it to where a simple transform as for the uniform B-site  case more work then simply rewriting them as new cells.

        Cells =  [get_cell_1RSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2),
                  get_cell_2RSL_IP(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_1LSL_IP(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_2RSL_IP(a, b, c, d1, d2, -alpha1, beta1, gamma1, -alpha2, beta2, gamma2),
                  get_cell_2RSL_IP(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2),
                  get_cell_1RSL_IP(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_2LSL_IP(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_1RSL_IP(a, b, c, d1, d2, -alpha1, beta1, gamma1, -alpha2, beta2, gamma2),
                  get_cell_2LSL_IP(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_1RSL_IP(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_1LSL_IP(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_2RSL_IP(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2)]            
    elif Rot_type == 'Out-of-plane':
        Cells =  [get_cell_1RSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2),
                  get_cell_2RSL(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_1LSL(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_2RSL(a, b, c, d1, d2, -alpha1, beta1, gamma1, -alpha2, beta2, gamma2),
                  get_cell_2RSL(a, b, c, d1, d2, alpha1, beta1, gamma1, alpha2, beta2, gamma2),
                  get_cell_1RSL(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_2LSL(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_1RSL(a, b, c, d1, d2, -alpha1, beta1, gamma1, -alpha2, beta2, gamma2),
                  get_cell_2LSL(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_1RSL(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2),
                  get_cell_1LSL(a, b, c, d1, d2, alpha1, -beta1, gamma1, alpha2, -beta2, gamma2),
                  get_cell_2RSL(a, b, c, d1, d2, alpha1, beta1, -gamma1, alpha2, beta2, -gamma2)]
    else:
        sys.exit('Unreconized rotation type. Choose \'In-plane\' or \'Out-of-plane\', for (001) oriented psuedocubic unit cell.')
        
    I = np.zeros(H.shape, dtype='complex128')
    for cell, vol_frac in zip(Cells, [D_1Rppp, D_2Rpmp, D_1Lppm, D_2Rmpp,   D_2Rppp, D_1Rpmp, D_2Lppm, D_1Rmpp]):
        F_HKL = calc_FSL(H, K, L, cell, a,b,c, symbols)
        L_P = LorenP(H, K, L, a,b, c, wavelength) + 1e-16*1j
        I += I0 / sind(eta) * L_P * F_HKL * np.conj(F_HKL)*vol_frac
    return np.real(I)

def set_rotation_domains(params,Rotations):
    if Rotations == 'Out-of-plane':
        params['D_2Lpmp'].vary =  False
        params['D_2Lpmp'].value =  0
        params['D_1Rppm'].vary =  False
        params['D_1Rppm'].value =  0
        params['D_1Lpmp'].vary =  False
        params['D_1Lpmp'].value =  0
        params['D_2Rppm'].vary =  False
        params['D_2Rppm'].value =  0
    elif Rotations == 'In-plane':
        params['D_2Rpmp'].vary =  False
        params['D_2Rpmp'].value =  0
        params['D_1Lppm'].vary =  False
        params['D_1Lppm'].value =  0
        params['D_1Rpmp'].vary =  False
        params['D_1Rpmp'].value =  0
        params['D_2Lppm'].vary =  False
        params['D_2Lppm'].value =  0
    else:
         sys.exit('Unreconized rotation type. Choose \'In-plane\' or \'Out-of-plane\', for (001) oriented psuedocubic unit cell.')               
    return params

def orthorhombic_only(params,Rotations):
    if Rotations == 'Out-of-plane':
        params['D_2Rppp'].vary =  False
        params['D_2Rppp'].value =  0
        params['D_1Rpmp'].vary =  False
        params['D_1Rpmp'].value =  0
        params['D_2Lppm'].vary =  False
        params['D_2Lppm'].value =  0
        params['D_1Rmpp'].vary =  False
        params['D_1Rmpp'].value =  0
    elif Rotations == 'In-plane':
        params['D_1Rppp'].vary =  False
        params['D_1Rppp'].value =  0
        params['D_1Lpmp'].vary =  False
        params['D_1Lpmp'].value =  0
        params['D_2Rppm'].vary =  False
        params['D_2Rppm'].value =  0
        params['D_2Rmpp'].vary =  False
        params['D_2Rmpp'].value =  0
    else:
         sys.exit('Unreconized rotation type. Choose \'In-plane\' or \'Out-of-plane\', for (001) oriented psuedocubic unit cell.')               
    return params

def setfix(params,ifix):
    ifix=np.array(ifix)
    ifix=ifix.astype(bool)
    ifix=np.logical_not(ifix)
    nlist = params.keys()
    for i in enumerate(nlist):
        params[i[1]].vary = ifix[i[0]]
    return params

def set_standard_constraintsSL(params):
    # Beta and  alpha locked together by strain
    params['beta1'].set(expr = 'alpha1')
    params['beta2'].set(expr = 'alpha2')


    nlist = params.keys()
    # Limit volume fractions to range 0 - 1
    for i in enumerate(nlist):
        if i[1][0] ==  'D':
            params[i[1]].set(min = 0, max = 1)
    #params['d1'].set(min = 0, max = .2)
    params['d1'].set(min = -0.2, max = .2)


    params['d2'].set(min = 0, max = .2)
    params['alpha1'].set(min = 0, max = 30)
    params['gamma1'].set(min = 0, max = 30)
    params['alpha2'].set(min = -30, max = 30)
    params['gamma2'].set(min = -30, max = 30)
    return params

