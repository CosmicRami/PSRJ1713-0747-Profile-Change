import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from astropy.time import Time
from scipy.stats import skew, kurtosis
from linpol_debias import debias_linpol
from PhaseGrad import get_offpulse_profile_rms, fft_profile_1d, fft_profile_2d, pgs_loss, get_phaseshifts
from scipy.optimize import curve_fit, minimize
import matplotlib.ticker as ticker

### Load Data ###
residuals_path = 'YOUR PATH'
files = sorted(os.listdir(residuals_path))
#print('List of files in directory have been captured')

mjds = []

scaling_factors_file = 'YOUR PATH / YOUR FILENAME TO SAVE SCALING FACTORS INTO'

with open(scaling_factors_file, 'a') as f:
    
    for filename in files:
        #print('Processing file:', filename)
        if filename.strip().endswith('YOUR FILES'):


            ### Template ###
            filename_primary = 'YOUR TEMPLATE FILE' 
            data_primary = np.loadtxt(filename_primary)
            bins_primary = data_primary[:, 2]
            I_profile_primary = data_primary[:, 3]
            Q_profile_primary = data_primary[:, 4]
            U_profile_primary = data_primary[:, 5]
            LinPol_profile_primary = debias_linpol(I_profile_primary, Q_profile_primary, U_profile_primary)
            V_profile_primary = data_primary[:, 6]
            PA_profile_primary = data_primary[:, 7]
            PAerr_profile_primary = data_primary[:, 8]

            ### Observation Files ###
            filename_secondary = os.path.join(residuals_path, filename)
            data_secondary = np.loadtxt(filename_secondary)
            #print(filename_secondary)
            bins_secondary = data_secondary[:, 2]
            I_profile_secondary = data_secondary[:, 3]
            Q_profile_secondary = data_secondary[:, 4]
            U_profile_secondary = data_secondary[:, 5]
	    LinPol_profile_secondary = debias_linpol(I_profile_secondary, Q_profile_secondary, U_profile_secondary)
            V_profile_secondary = data_secondary[:, 6]
            pa_secondary = data_secondary[:, 7]
            pa_err_secondary = data_secondary[:, 8]


            ### Normalization ###
            def integrate_profile(profile, start_bin, end_bin):
                return np.sum(profile[start_bin:end_bin])

            area_I_primary = integrate_profile(I_profile_primary, start_bin, end_bin)
            area_I_secondary = integrate_profile(I_profile_secondary, start_bin, end_bin)

            I_profile_primary_normalized = I_profile_primary / area_I_primary
            LinPol_profile_primary_normalized = LinPol_profile_primary / area_I_primary 
            V_profile_primary_normalized = V_profile_primary / area_I_primary 

            I_profile_secondary_normalized = I_profile_secondary / area_I_secondary
            LinPol_profile_secondary_normalized = LinPol_profile_secondary / area_I_secondary 
            V_profile_secondary_normalized = V_profile_secondary / area_I_secondary 

            ### Phase Gradient FFT Shift ###
            def fourier_roll(array, shift):
                N = len(array)
                x_array = np.arange(-N//2, N//2)
                phaseshift = np.exp(-2.0 * np.pi * 1j * shift * x_array / N)
                array_fft = np.fft.fftshift(np.fft.fft(array))
                array_fft_shifted = phaseshift * array_fft
                array_shifted = np.fft.ifft(np.fft.ifftshift(array_fft_shifted)).real
                return array_shifted

            sigmas_primary_I = get_offpulse_profile_rms(I_profile_primary_normalized, 0, 1023, int(0.3 * 1023), int(0.7 * 1023))
            sigmas_secondary_I = get_offpulse_profile_rms(I_profile_secondary_normalized, 0, 1023, int(0.3 * 1023), int(0.7 * 1023))

            fft_primary, fft_primary_amp, fft_primary_phase = fft_profile_1d(I_profile_primary_normalized)
            fft_secondary, fft_secondary_amp, fft_secondary_phase = fft_profile_1d(I_profile_secondary_normalized)

            phaseshifts, phaseshifts_err, bs, hess_inv = get_phaseshifts(
                fft_secondary_amp, fft_secondary_phase, sigmas_secondary_I,
                fft_primary_amp, fft_primary_phase, nfftfreqs=25
            )

            N = len(I_profile_secondary_normalized)
            shift = phaseshifts[0] / (2.0 * np.pi) * N

            I_profile_secondary_shifted = fourier_roll(I_profile_secondary_normalized, shift)
            LinPol_profile_secondary_shifted = fourier_roll(LinPol_profile_secondary_normalized, shift)
            V_profile_secondary_shifted = fourier_roll(V_profile_secondary_normalized, shift)


            ### Scaling ###
            def objective(scaling_factors):
                scaled_I_profile_secondary = scaling_factors[0] * I_profile_secondary_shifted[462:562]
                scaled_LinPol_profile_secondary = scaling_factors[1] * LinPol_profile_secondary_shifted[462:562]
                scaled_V_profile_secondary = scaling_factors[2] * V_profile_secondary_shifted[462:562]

                return np.concatenate([
                    I_profile_primary_normalized[462:562] - scaled_I_profile_secondary,
                    LinPol_profile_primary_normalized[462:562] - scaled_LinPol_profile_secondary,
                    V_profile_primary_normalized[462:562] - scaled_V_profile_secondary,
                ])

            scaling_factors_initial = [0.1, 0.1, 0.1]
            scaling_factors_optimised = least_squares(objective, scaling_factors_initial).x
            #print('The optimised scaling factors for the I, LinPol, and V profiles are:', scaling_factors_optimised)

            I_profile_secondary_downsampled_scaled = scaling_factors_optimised[0] * I_profile_secondary_shifted
            LinPol_profile_secondary_downsampled_scaled = scaling_factors_optimised[0] * LinPol_profile_secondary_shifted
            V_profile_secondary_downsampled_scaled = scaling_factors_optimised[0] * V_profile_secondary_shifted
            
            f.write(f"{filename}  {scaling_factors_optimised[0]:.6f}\n")

            ### Profile Residuals ###
            opt_residuals_I = I_profile_secondary_downsampled_scaled - I_profile_primary_normalized
            opt_residuals_LinPol = LinPol_profile_secondary_downsampled_scaled - LinPol_profile_primary_normalized
            opt_residuals_V = V_profile_secondary_downsampled_scaled - V_profile_primary_normalized


            ### Write Residual Files ###

            prefix = filename[:10]
            date_str_ = prefix.split('_')[1]
            date_str = "20" + date_str_
            date_str_2 = date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:8] + ' 00:00:00'
            date = Time(date_str_2, format = 'iso')
            date_mjd = date.mjd
            mjds.append(date_mjd)

            residuals_path = "YOUR PATH"

            with open(residuals_path + prefix + "_I_profile_residuals2.txt", "w") as file:
                file.write("\n".join(map(str, opt_residuals_I)))

            with open(residuals_path + prefix + "_LinPol_profile_residuals2.txt", "w") as file:
                file.write("\n".join(map(str, opt_residuals_LinPol)))

            with open(residuals_path + prefix + "_V_profile_residuals2.txt", "w") as file:
                file.write("\n".join(map(str, opt_residuals_V)))
        
print('Loop has finished running')

### Making NP Arrays of Profile Residuals ###

residuals_I_profile_list = []
datelist = []
for filename2 in files:
    
    if filename2.endswith('_I_profile_residuals2.txt'):
        data = np.loadtxt(filename2)
        residuals_I_profile_list.append(data)

residuals_I_profile_array2 = np.array(residuals_I_profile_list)

output_filename2 = "residuals_I_profile_array2.txt"
np.savetxt(output_filename2, residuals_I_profile_array2)

print(f"Saved the residuals2 I profile array to {output_filename2}")

residuals_LinPol_profile_list = []

for filename3 in files:
    if filename3.endswith('_LinPol_profile_residuals2.txt'):
        data = np.loadtxt(filename3)
        residuals_LinPol_profile_list.append(data)

residuals_LinPol_profile_array2 = np.array(residuals_LinPol_profile_list)

output_filename3 = "residuals_LinPol_profile_array2.txt"
np.savetxt(output_filename3, residuals_LinPol_profile_array2)

print(f"Saved the residuals2 LinPol profile array to {output_filename3}")

residuals_V_profile_list = []

for filename5 in files:
    if filename5.endswith('_V_profile_residuals2.txt'):
        data = np.loadtxt(filename5)
        residuals_V_profile_list.append(data)

residuals_V_profile_array2 = np.array(residuals_V_profile_list)

output_filename5 = "residuals_V_profile_array2.txt"
np.savetxt(output_filename5, residuals_V_profile_array2)

print(f"Saved the residuals2 V profile array to {output_filename5}")


### Building MJD array ###

cadence = np.diff(mjds)
time_order = np.argsort(mjds)
residuals_I_profile_array = residuals_I_profile_array2[time_order, :]
residuals_LinPol_profile_array = residuals_LinPol_profile_array2[time_order, :]
residuals_V_profile_array = residuals_V_profile_array2[time_order, :]

mjds_ordered = np.array(mjds)[time_order]
mjd_arr = np.arange(np.min(mjds_ordered), np.max(mjds_ordered)+1, 1)
mjd_arr
residuals_I_profile_plot = []
residuals_LinPol_profile_plot = []
residuals_V_profile_plot = []

ind_ = 0

for i__, mjd_ in enumerate(mjd_arr):
    if mjd_ in mjds_ordered and mjd_ > mjds_ordered[0]:
        ind_ += 1
    residuals_I_profile_plot.append(residuals_I_profile_array[ind_])
    residuals_LinPol_profile_plot.append(residuals_LinPol_profile_array[ind_])
    residuals_V_profile_plot.append(residuals_V_profile_array[ind_])

residuals_I_profile_plot = np.array(residuals_I_profile_plot)
residuals_LinPol_profile_plot = np.array(residuals_LinPol_profile_plot)
residuals_V_profile_plot = np.array(residuals_V_profile_plot)

print('I_profile shape:', residuals_I_profile_plot.shape)
print('L_profile shape:', residuals_LinPol_profile_plot.shape)
print('V_profile shape:', residuals_V_profile_plot.shape)

### Plotting ###

figure_width = 15
figure_height = 10

fig, axs = plt.subplots(1, 3, figsize=(figure_width, figure_height), sharey=True)

im1 = axs[0].imshow(residuals_I_profile_plot[:, 358:666], vmin=-0.006, vmax=0.006, aspect='auto', origin='lower', interpolation='none', extent = [-0.15, 0.15, mjds_ordered[0], mjds_ordered[-1]])
axs[0].set_title("I_Profile Residuals", fontsize=16)
axs[0].set_xlabel("Pulse Phase", fontsize=16)
axs[0].set_ylabel("MJD", fontsize=16) # Extra line in the first plot for y-axis ticks and labels
axs[0].axhline(y=59320, color='red', linestyle='--')


im2 = axs[1].imshow(residuals_LinPol_profile_plot[:, 358:666], vmin=-0.006, vmax=0.006, aspect='auto', origin='lower', interpolation='none',  extent = [-0.15, 0.15, mjds_ordered[0], mjds_ordered[-1]])
axs[1].set_title("Linear Polarisation Profile Residuals", fontsize=16)
axs[1].set_xlabel("Pulse Phase", fontsize=16)
axs[1].axhline(y=59320, color='red', linestyle='--')

im3 = axs[2].imshow(residuals_V_profile_plot[:, 358:666], vmin=-0.006, vmax=0.006, aspect='auto', origin='lower', interpolation='none', extent = [-0.15, 0.15, mjds_ordered[0], mjds_ordered[-1]])
axs[2].set_title("V_Profile Residuals", fontsize=16)
axs[2].set_xlabel("Pulse Phase", fontsize=16)
axs[2].axhline(y=59320, color='red', linestyle='--')

fig.suptitle("J1713+0747 - I, L, V Profile Residuals: <Freq. Range> MHz", fontsize=22, y=0.95)

cax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
fig.colorbar(im3, cax=cax)

for ax in axs:
    ax.tick_params(axis='y', labelsize=12) 

#plt.savefig('test_no_initial_Center_allStokes_align.png', dpi = 300)

plt.show()
        
        