import numpy as np

def update_fres(fs, zs, fres, spans, fcal_indices, method = 'distance'):
    """
    Update resonance frequencies given fine sweep data

    Parameters:
    fs (array-like): fine sweep frequency data in Hz for each resonator in fres
    zs (array-like): fine sweep complex S21 data for each resonator in fres
    fcal_indices (array-like): list of calibrations tone indices (index into
        fs, zs, fres, Qres). Calibration tone frequencies will not be updated
    method (str): 'mins21' to update using the minimum of |S21|. 'spacing' to
        update using the maximum spacing between adjacent IQ points. 'distance'
        to update using the point of furthest distance from the off-resonance
        point. 'none' to return fres.
    fres (np.array or None): list of resonance frequencies in Hz
    Qres (np.array or None): list of quality factors to cut if
        cut_other_resonators, or None. Cuts spans of fres / Qres from each
        sweep

    Returns:
    fres_new (np.array): array of updated frequencies in Hz
    """
    if method == 'none':
        fres = []
        for i in range(len(f) // npoints):
            i0, i1 = npoints * i, npoints * (i + 1)
            fi = f[i0:i1]
            fres.append(np.mean(fi))
        return np.array(fres)
    elif method == 'mins21':
        update = update_fr_minS21
    elif method == 'spacing':
        update = update_fr_spacing
    elif method == 'distance':
        update = update_fr_distance
    else:
        raise ValueError("method must be 'mins21', 'distance', or 'spacing'")
    fres_new = []
    for i in range(len(f) // npoints):
        i0, i1 = npoints * i, npoints * (i + 1)
        fi, zi = f[i0:i1], z[i0:i1]
        if i not in fcal_indices:
            if cut_other_resonators:
                spans = fres / Qres
                fi, zi = cut_fine_scan(fi, zi, fres, spans)
            fres_new.append(update(fi, zi))
        else:
            fres_new.append(np.mean(fi))
    return np.array(fres_new)

def update_fr_minS21(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the minimum of |S21| with a linear fit subtracted

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    dB = 20 * np.log10(abs(z))
    dB0 = dB - np.polyval(np.polyfit(f, dB, 1), f)
    ix = np.argmin(dB0)
    fr = f[ix]
    return fr

def update_fr_spacing(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the max spacing between adjacent IQ points

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    spacing = np.abs(np.diff(z))
    spacing = spacing[1:] + spacing[:-1]
    spacing = np.concatenate([[0],spacing, [0]])
    ix = np.argmax(spacing)
    fr = f[ix]
    return fr

def update_fr_distance(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the furthest point from the off-resonance data. This
    function will perform better if the cable delay is first removed.

    Parameters:
    f (np.array): Single resonator frequency data
    z (np.array): Single resonator complex S21 data

    Returns:
    fr (float): Updated frequency
    """
    offres = np.mean(list(z[:10]) + list(z[-10:]))
    diff = abs(z - offres)
    ix = np.argmax(diff)
    fr = f[ix]
    return fr

def cut_fine_scan(f, z, fres, spans):
    """
    Cuts resonance frequencies out of a single set of fine scan data

    Parameters:
    f, z (np.array, np.array): fine scan frequency in Hz and complex S21 data
    fres (np.array): array of frequencies to cut in Hz
    spans (np.array): array of frequency spans in Hz to cut
    """
    ix = (fres <= max(f)) & (fres >= min(f)) & (np.abs(fres - np.mean(f)) > 1e3)
    fres, spans = fres[ix], spans[ix]
    for fr, sp in zip(fres, spans):
        ix = abs(f - fr) > sp
        f, z = f[ix], z[ix]
    return f, z
