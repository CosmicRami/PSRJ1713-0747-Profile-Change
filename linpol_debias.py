import numpy as np



def debias_linpol(I, Q, U):

    """
    routine to de-bias the linear polarization
    Follows Muller, Beck, Krause (2017)
    
    """
    
    I__ = I.copy()
    sigma_I = 0.0
    
    for i in range(8):
        sigma_I = np.std(I__)
        I__ = I__[I__ < 1.57*sigma_I]
        #print(sigma_I)
    sigma_I = np.std(I__)

    L = np.sqrt(Q**2.0 + U**2.0)
    
    theta_meas = np.arctan2(U, Q)
    x_meas = np.cos(theta_meas)
    y_meas = np.sin(theta_meas)
    
    U_refl = np.concatenate([U, U, U])
    Q_refl = np.concatenate([Q, Q, Q])
    
    theta_refl = np.arctan2(U_refl,Q_refl)
    x = np.cos(theta_refl)
    y = np.sin(theta_refl)
    
    phasebins = np.arange(len(I), len(I) + len(I), 1, dtype="int")
    
    xm = []
    ym = []
    xmm = []
    ymm = []
    for phasebin in phasebins:
        
        xm.append(np.median(x[phasebin-5:phasebin+5]))
        ym.append(np.median(y[phasebin-5:phasebin+5]))
        
        xmm.append(np.median(np.concatenate([x[phasebin-5:phasebin-1], x[phasebin+1:phasebin+5]])))
        ymm.append(np.median(np.concatenate([y[phasebin-5:phasebin-1], y[phasebin+1:phasebin+5]])))
        #xmm_[phasebin] -= x[phasebin]

    xm = np.array(xm)
    ym = np.array(ym)
    theta_m = np.arctan2(ym, xm)

    xmm = np.array(xmm)
    ymm = np.array(ymm)
    theta_mm = np.arctan2(ymm, xmm)

    Pm = U * np.sin(theta_m) + Q * np.cos(theta_m)
    Pmm = U * np.sin(theta_mm) + Q * np.cos(theta_mm)
    P_debias = 1.0*(1.0*Pm + 2.0*Pmm)/3

    return(P_debias)


