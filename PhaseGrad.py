import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt


def get_offpulse_profile_rms(profile, offpulse_ind0, offpulse_ind1, fft_ind0, fft_ind1): 
    
    if len(profile.shape) > 1:
        sigmas = np.std(np.abs(np.fft.fft(np.abs(profile)[offpulse_ind0:offpulse_ind1])[fft_ind0:fft_ind1, :]), axis = 0)
        
    else:
        sigmas = np.array([np.abs(np.fft.fft(np.abs(profile)[offpulse_ind0:offpulse_ind1])[fft_ind0:fft_ind1])])
    
    return sigmas.squeeze() #returns off-pulse rms


def fft_profile_1d(profile):
    
    fft_ = np.fft.fft(np.abs(profile)) 
    fft_amp_ = np.abs(fft_) 
    fft_phase_ = np.arctan2(np.imag(fft_), np.real(fft_)) 

    return(fft_, fft_amp_, fft_phase_) 

def fft_profile_2d(profile): 
    
    ffts_ = []
    ffts_amp_ = []
    ffts_phase_ = []

    for i in range(profile.shape[1]):
        fft_, fft_amp_, fft_phase_ = fft_profile_1d(profile[:, i]) 
        ffts_.append(fft_) 
        ffts_amp_.append(fft_amp_) 
        ffts_phase_.append(fft_phase_) 
    ffts_ = np.array(ffts_).T 
    ffts_amp_ = np.array(ffts_amp_).T 
    ffts_phase = np.array(ffts_phase_).T 

    return(ffts_, ffts_amp_, ffts_phase_) 

def pgs_loss(x, Pk, thetak, sigma, Sk, phik):
    
    #x = the parameters to fit for
    #Pk, thetak = Fourier amplitudes and phases for the profile FFT
    #sigma = off-pulse rms on the profile
    #Sk, phik = Fourier amplitudes and phases of the template FFT
    
    b, tau = x 
    
    chi2 = sigma**-2 * np.sum(Pk**2.0 + b**2.0 * Sk**2.0) \
            - 2.0 * b * sigma**-2.0 * np.sum(Pk * Sk * np.cos(phik - thetak + np.arange(len(Pk))*tau))
    #sigma^2 x sum(Obs_amp^2 + b^2 + Temp_amp^2) \ -2 x b x sigma^2 x sum(obs_amp * temp_am x np.cos(temp_phase - obs_phase))

    return(np.log10(chi2))



def get_phaseshifts(fft_profs_amp, fft_profs_phase, sigmas, fft_templ_amps, fft_templ_phase, nfftfreqs=16):
    
    phaseshifts = []
    phaseshifts_err = []
    bs = [] 
    toas = [] 
    hess_inv = [] 
    
    
    if len(fft_profs_amp.shape) == 1:
        fft_profs_amp = fft_profs_amp.reshape(-1, 1)
        fft_profs_phase = fft_profs_phase.reshape(-1, 1)
        fft_templ_amps = fft_templ_amps.reshape(-1, 1)
        fft_templ_phase = fft_templ_phase.reshape(-1, 1)
    
    #print(fft_profs_amp.shape)
    
    N = (fft_profs_amp.shape[0])
    #print(N)
    
    for i in range(fft_profs_amp.shape[1]):
#         print(fft_templ_phase[:nfftfreqs])
#         plt.plot((fft_profs_phase[:, i][:nfftfreqs].squeeze() + fft_templ_phase[:nfftfreqs].squeeze())%(2.0*np.pi)-np.pi,'x')
#         plt.show()
        #plt.plot((fft_profs_phase[:, i])[:nfftfreqs], 'x')
        #plt.show()
        #[N//2 - nfftfreqs:N//2 + nfftfreqs]
        
        result = minimize(pgs_loss, (0.1, 0.0), args=((fft_profs_amp[:, i].squeeze())[:nfftfreqs].squeeze(),
                                                       (fft_profs_phase[:, i].squeeze())[:nfftfreqs].squeeze(),
                                                       sigmas[i],
                                                       (fft_templ_amps.squeeze())[:nfftfreqs].squeeze(),
                                                       (fft_templ_phase.squeeze())[:nfftfreqs].squeeze()),
                                                       options={"maxiter": 1000})
        
        
        phaseshifts.append(result.x[1])
        bs.append(result.x[0])
        # toas.append(t.tdb[0].mjd*86400 + (i)* P0 + coeff[1]*P0 + P0*result.x[1]/(2.0*np.pi))
        hess_inv.append(result.hess_inv)
        phaseshifts_err.append((0.5 * result.hess_inv[1, 1]) ** 0.5)
    
    
    phaseshifts = np.array(phaseshifts)
    phaseshifts_err = np.array(phaseshifts_err)
    bs = np.array(bs)
    hess_inv = np.array(hess_inv)

    return phaseshifts, phaseshifts_err, bs, hess_inv


if __name__ == "__main__":
    
   
    offpulse_ind0 = 0
    offpulse_ind1 = 400
    fft_ind0  = int(0.3 * (offpulse_ind1 - offpulse_ind0))
    fft_ind1 = int(0.7 * (offpulse_ind1 - offpulse_ind0))
    sigmas = get_offpulse_profile_rms(profiles, offpulse_ind0, offpulse_ind1, fft_ind0, fft_ind1)
    fft_profs, fft_profs_amp, fft_profs_phase = fft_profile_2d(profiles)
    fft_templ, fft_templ_amps, fft_templ_phase = fft_profile_1d(template)
    get_phaseshifts(fft_profs_amp, fft_profs_phase, sigmas, fft_templ_amps, fft_templ_phase, nfftfreqs=40)
