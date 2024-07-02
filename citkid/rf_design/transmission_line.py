import numpy as np

eps0 = 8.854e-12
Z0vac = 376.73
c = 3e8
hbar = 1.055e-34
h = 6.626e-34
k_B = 1.381e-23

def get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq):
    '''
    Calculates transmission line parameters of a microstrip line, with the option
    to include a higher-permittivity material in place of the vacuum (i.e., the superstrate).
    All formulae are from "Accurate models for microstrip computer-aided design",
    doi: 10.1109/MWSYM.1980.1124303

    Note that it will return complex values if eps_sup > eps_sub, but the real parts
    still seem to agree well with Sonnet simulations.
    Z0_tot will generally have a complex part anyway if Rsq > 0.

    Parameters:
        eps_sub <float>: relative permittivity of substrate
        eps_sup <float>: relative permittivity of superstrate
        w_strip <float>: width of center line in m
        h_sub <float>: height of substrate in m
        Rsq <float>: sheet resistance of center line in Ohms/sq
        freq <float>: frequency in Hz
    Returns:
        eps_eff <float>: effective dielectric constant of approximate TEM mode
        L <float>: inductance in H/m
        C <float>: capacitance in F/m
        Z0_tot <float>: impedance of approximate TEM mode
    '''

    # effective permittivity of approximate TEM mode
    eps_rel = eps_sub/eps_sup 
    u = w_strip / h_sub
    a = 1 + 1/49*np.log((u**4+(u/52)**2)/(u**4+.432)) + 1/18.7*np.log(1+(u/18.1)**3)
    b = 0.564*((eps_rel-0.9)/(eps_rel+3))**.053
    eps_eff = (eps_rel+1)/2 + (eps_rel-1)/2 * (1+10/u)**(-a*b)
    eps_eff *= eps_sup

    # impedance of approximate TEM mode
    f = 6 + (2*np.pi-6)*np.exp(-(30.666/u)**.7528)
    Z0 = Z0vac/(2*np.pi*np.sqrt(eps_eff)) * np.log(f/u + np.sqrt(1+(2/u)**2))
    
    # impedance and capacitance per length
    vph = c/np.sqrt(eps_eff)
    L = Z0/vph
    C = 1/(Z0*vph)
    
    # impedance after taking into account sheet resistance
    R = Rsq/w_strip
    omega = 2*np.pi*freq
    Z0_tot = np.sqrt((R+1j*omega*L)/(1j*omega*C))
    
    return eps_eff, L, C, Z0_tot

def get_sc_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq, Tc):
    '''
    Same as get_mstrip_params(), with the addition of kinetic inductance.
    Assumes a thin superconducting film (i.e. w_strip << penetration depth).
    Kinetic inductance is calculated using eq. (47) of JZ12 (10.1146/annurev-conmatphys-020911-125022)

    Parameters:
        see get_mstrip_params()
        Tc <float>: superconducting critical temperature in K
    Returns:
        see get_mstrip_params()
        Lkin <float>: kinetic inductance in H/m
    '''
    eps_eff, L, C, Z0_tot = get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq)
    Delta0 = 1.76*k_B*Tc
    Lkin = hbar*Rsq/(np.pi*Delta0*w_strip)
    Ltot = L + Lkin

    R = Rsq/w_strip
    omega = 2*np.pi*freq
    Z0_tot = np.sqrt((R+1j*omega*Ltot)/(1j*omega*C))
    
    return eps_eff, L, Lkin, C, Z0_tot