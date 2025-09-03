import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from linpol_debias import debias_linpol
from PhaseGrad import get_offpulse_profile_rms, fft_profile_1d, fft_profile_2d, pgs_loss, get_phaseshifts

### Generate S/N listing of files ###
file_list = [filename for filename in os.listdir() if filename.endswith("YOUR FILES")]

stokes_profiles = {'I': [], 'Q': [], 'U': [], 'V': [], 'L': []}
pa_profiles = {'PA': [], 'PAerr': [], 'PA2': []}

fig = plt.figure()

for filename in file_list:
    data = np.loadtxt(filename)
    bins_primary = data[:, 2]
    I_profile = data[:, 3]
    Q_profile = data[:, 4]
    U_profile = data[:, 5]
    V_profile = data[:, 6]
    PA_profile = data[:, 7]
    PAerr_profile = data[:, 8]
    PA_profile_2 = 0.5 * np.arctan2(U_profile, Q_profile) 
          
    PA_profile[PA_profile == 0] = np.nan
    PAerr_profile[PAerr_profile == 0] = np.nan
    PA_profile_2[PA_profile == 0] = np.nan
    
    LinPol_profile = debias_linpol(I_profile, Q_profile, U_profile)
    
    plt.plot(I_profile, c = 'k', alpha = 0.1) 
       
    stokes_profiles['I'].append(I_profile)
    stokes_profiles['Q'].append(Q_profile)
    stokes_profiles['U'].append(U_profile)
    stokes_profiles['L'].append(LinPol_profile)
    stokes_profiles['V'].append(V_profile)
    pa_profiles['PA'].append(PA_profile)
    pa_profiles['PAerr'].append(PAerr_profile)
    pa_profiles['PA2'].append(PA_profile_2)
    
plt.show() 

sn_data = np.loadtxt('PreEvent_S2N.txt', dtype={'names': ('filename', 'sn'), 'formats': ('U40', 'f4')})
sn_sorted = np.sort(sn_data, order='sn')[::-1]

print("Sorted files by S/N:") 

for item in sn_sorted:
    print(f"{item['filename']} {item['sn']}")


formatted_filenames = [f"{item['filename'][:10]}_sample.txt" for item in sn_sorted]
sorted_files_by_sn = formatted_filenames


### Generate Templates ###
file_list = sorted_files_by_sn  

def fourier_roll(array, shift):  

    N = len(array)
    x_array = np.arange(-N//2, N//2)
    phaseshift = np.exp(-2.0 * np.pi * 1j * shift * x_array / N)
    array_fft = np.fft.fftshift(np.fft.fft(array))
    array_fft_shifted = phaseshift * array_fft
    array_shifted = np.fft.ifft(np.fft.ifftshift(array_fft_shifted)).real
    return array_shifted

def load_and_process(filename):

    data = np.loadtxt(filename)
    bins = data[:, 2]
    I = data[:, 3]
    Q = data[:, 4]
    U = data[:, 5]
    V = data[:, 6]
    PA = data[:, 7]
    PAerr = data[:, 8]
    
    PA[PA == 0] = np.nan
    PAerr[PAerr == 0] = np.nan 
    
    return bins, I, Q, U, V, PA, PAerr

bins, dynamic_template_I, dynamic_template_Q, dynamic_template_U, dynamic_template_V, dynamic_template_PA, dynamic_template_PAerr = load_and_process(file_list[0])

accumulated_profiles = {'I': [dynamic_template_I], 'Q': [dynamic_template_Q], 'U': [dynamic_template_U], 
                        'V': [dynamic_template_V], 'PA': [dynamic_template_PA], 'PAerr': [dynamic_template_PAerr]}

for filename in file_list[1:]:
    bins, I, Q, U, V, PA, PAerr = load_and_process(filename)
    
    sigmas_I = get_offpulse_profile_rms(I, 0, 1023, int(0.3 * 1023), int(0.7 * 1023))  
    fft_template, fft_template_amp, fft_template_phase = fft_profile_1d(dynamic_template_I)  
    fft_dynamic_profile, fft_dynamic_profile_amp, fft_dynamic_profile_phase = fft_profile_1d(I)  
    
    phaseshifts, _, _, _ = get_phaseshifts(
        fft_dynamic_profile_amp, fft_dynamic_profile_phase, sigmas_I,
        fft_template_amp, fft_template_phase, nfftfreqs=25
    )
    
    shift_I = phaseshifts[0] / (2.0 * np.pi) * len(I)  
    
    I = fourier_roll(I, shift_I)
    Q = fourier_roll(Q, shift_I)
    U = fourier_roll(U, shift_I)
    V = fourier_roll(V, shift_I)
    PA = fourier_roll(PA, shift_I)
    PAerr = fourier_roll(PAerr, shift_I)
    
    PA[PA == 0] = np.nan
    PAerr[PAerr == 0] = np.nan
    
    accumulated_profiles['I'].append(I)
    accumulated_profiles['Q'].append(Q)
    accumulated_profiles['U'].append(U)
    accumulated_profiles['V'].append(V)
    accumulated_profiles['PA'].append(PA)
    accumulated_profiles['PAerr'].append(PAerr)
    
    dynamic_template_I = np.mean(accumulated_profiles['I'], axis=0)
    dynamic_template_Q = np.mean(accumulated_profiles['Q'], axis=0)
    dynamic_template_U = np.mean(accumulated_profiles['U'], axis=0)
    dynamic_template_V = np.mean(accumulated_profiles['V'], axis=0)

    valid_PA = ~np.isnan(PA) & ~np.isnan(dynamic_template_PA)
    valid_PAerr = ~np.isnan(PAerr) & ~np.isnan(dynamic_template_PAerr)
    
    dynamic_template_PA[valid_PA] = np.mean(accumulated_profiles['PA'], axis=0)[valid_PA]
    dynamic_template_PAerr[valid_PAerr] = np.mean(accumulated_profiles['PAerr'], axis=0)[valid_PAerr]


np.savetxt("YOUR TEMPLATE FILE", np.column_stack((bins, dynamic_template_I, dynamic_template_Q, dynamic_template_U, dynamic_template_V, dynamic_template_PA, dynamic_template_PAerr)), fmt='%f', header="Bins I Q U V PA PAerr")

print('Final template created and saved')

### Plotting Results ###

#Stokes I--------------------------------------------------------------------------------------------
final_template_data = np.loadtxt("YOUR TEMPLATE FILE")
bins_final, I_final = final_template_data[:, 0], final_template_data[:, 1]

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Stokes I Profiles with Final Average', fontsize=16)
ax.set_xlabel('Pulse Phase', fontsize=14)
ax.set_ylabel('Flux Density (mJy)', fontsize=14)
ax.grid(True)

for filename in sorted_files_by_sn:

    data = np.loadtxt(filename)
    bins = data[:, 2]
    I_profile = data[:, 3]
    ax.plot(bins, I_profile, 'k', alpha=0.3)  

ax.plot(bins_final, I_final, 'r-', label='Final Averaged Template', linewidth=2)
ax.legend()
#ax.set_xlim(358, 666)
plt.show()


#Stokes V-------------------------------------------------------------------------------------------
final_template_data = np.loadtxt("YOUR TEMPLATE FILE")
bins_final, V_final = final_template_data[:, 0], final_template_data[:, 4]  


fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Stokes V Profiles with Final Average', fontsize=16)
ax.set_xlabel('Pulse Phase', fontsize=14)
ax.set_ylabel('Flux Density (mJy)', fontsize=14)
ax.grid(True)

for filename in sorted_files_by_sn:
    data = np.loadtxt(filename)
    bins = data[:, 2]
    V_profile = data[:, 6]  
    ax.plot(bins, V_profile, 'k', alpha=0.3)  

ax.plot(bins_final, V_final, 'r-', label='Final Averaged Template', linewidth=2)
ax.legend()
#ax.set_xlim(358, 666)
plt.show()


#PA-----------------------------------------------------------------------------------------------------
final_template_data = np.loadtxt("YOUR TEMPLATE FILE")
bins_final, PA_final = final_template_data[:, 0], final_template_data[:, 5]  


fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title('Position Angle with Final Average', fontsize=16)
ax.set_xlabel('Pulse Phase', fontsize=14)
ax.set_ylabel('Degrees', fontsize=14)
ax.grid(True)

for filename in sorted_files_by_sn:
    data = np.loadtxt(filename)
    bins = data[:, 2]
    PA_profile = data[:, 7]
    PA_profile[PA_profile == 0] = np.nan
    ax.scatter(bins, PA_profile, color='k', alpha=0.3)  

ax.scatter(bins_final, PA_final, color='r', label='Final Averaged Template', linewidth=2)
ax.legend()
ax.set_xlim(358, 666)
ax.set_ylim(-90, 90)
plt.show()


# All STokes -----------------------------------------------------------------------------------------
data = np.loadtxt("YOUR TEMPLATE FILE")
bins = data[:, 0]
I = data[:, 1]
Q = data[:, 2]
U = data[:, 3]
L = debias_linpol(I, Q, U)
V = data[:, 4]

plt.figure(figsize=(15, 10))
plt.plot(bins, I, label='Stokes I', color='black')
plt.plot(bins, Q, label='Stokes Q', color='brown')
plt.plot(bins, U, label='Stokes U', color='orange')
plt.plot(bins, L, label='LinPol', color='red')
plt.plot(bins, V, label='Stokes V', color='blue')
plt.title('S/N Optimized FFT-Aligned Template', fontsize=16)
plt.xlabel('Pulse Phase', fontsize=14)
plt.ylabel('Flux Density (mJy)', fontsize=14)
plt.legend()
plt.grid(True)
#plt.xlim(358, 666)
plt.show()
