# %% [markdown]
# # Library

# %%
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

# from astropy.cosmology import WMAP9 as cosmo
# from astropy.constants import c
# from astropy import units as u
# from astropy.cosmology import z_at_value
import inspect

import sys
sys.path.append('..')
from util.helper import *
from util.sdtpy import *
register_custom_filters_on_speclite('../util')

# %% [markdown]
# # Setting

# %% [markdown]
# - Define `speclite` object

# %%
intype = 'kn'
indist = 40
inexptime = 180
group = input("""Choose the filterset (med25nm, broad_griz, broad_ugriz):""")
# group = 'broad_ugriz'
if group == 'med25nm':
	filterset_group = filterlist_med25nm
elif group == 'broad_griz':
	filterset_group = filterlist_griz
elif group == 'broad_ugriz':
	filterset_group = filterlist_ugriz
filterset = [f"{group}-{filte}" for filte, _, group in filterset_group]
bands = speclite.filters.load_filters(*filterset)

# %% [markdown]
# ## Kilonova
# - GW170817-like morphology model

# %%
import pickle
path_pickle = '../model/PLAsTiCC/SNII-NMF/SNII.pickle'
with open(path_pickle, 'rb') as f:
	interp = pickle.load(f)
# interp = interp_gw170817like_kn(path_kntable='/Users/paek/Research/7DT/7dtpy/3.table/kn_sim_cube_lite', path_pickle=None)
# lamarr = np.arange(1003, 127695.+1, 1)

# %%
if 'med' in bands.names[0]:
	#    Medium-band
	bandwidth = 250 # [AA]
	lammin = 3000
	lammax = 10000
else:
	#	Broad-band
	bandwidth = 1000 # [AA]
	lammin = 3000
	lammax = 12000

# lammin = bands.effective_wavelengths.min().value-bandwidth*2
# lammax = bands.effective_wavelengths.max().value+bandwidth*2

lamstep = bandwidth/10
lamarr = np.arange(lammin, lammax+lamstep, lamstep)
_lamarr = np.arange(lammin, lammax+lamstep, 10)

print(f"lam: {lammin:.3f} - {lammax:.3f} AA")
print(f"lamstep: {lamstep:g} AA")
print(f"n_lam: {len(lamarr)}")

#	Redshift
zarr = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])

# %% [markdown]
# - Function to fit
# %%
def func(x, z, t, pc1, pc2, pc3):

	z0 = 0
	#	Input
	point = (
		pc1,
		pc2,
		pc3,
		t,
		lamarr,
	)

	iflux = interp(point)
	(zspappflam, zsplam) = apply_redshift_on_spectrum(iflux*flamunit, lamarr*lamunit, z, z0)
	mags = bands.get_ab_magnitudes(*bands.pad_spectrum(zspappflam, zsplam))

	spmag = np.array([mags[key][0] for key in mags.keys()])
	spfnu = (spmag*u.ABmag).to(u.uJy).value

	return spfnu

# %% [markdown]
# ## Variables to set

# %%
# %%
snrcut = 3
source = 'MOSFiT'
fittype = 'SNII-NMF'
verbose = True

path_model = f'../model/PLAsTiCC/{fittype}/SIMSED.{fittype}'
models = sorted(glob.glob(f'{path_model}/*.fits'))
print(f"{len(models)} models found") 

infotbl = ascii.read(f'../model/PLAsTiCC/{fittype}/sedinfo.dat')
infotbl['model'] = [f"{path_model}/{model}" for model in infotbl['model']] 


ii = 10
model = models[ii]
_mdtbl = Table.read(model)

indx = np.where(
    (_mdtbl['col1'] <= 30) &
    (_mdtbl['col2'] >= 2000) &
    (_mdtbl['col2'] <= 11000)
    # (_mdtbl['col2'] <= 10000)
)

mdtbl = _mdtbl[indx]
phasearr = np.unique(mdtbl['col1'])
lamarr = np.unique(mdtbl['col2'])
number_of_unique_phase, number_of_unique_wavelength = len(phasearr), len(lamarr)
flux2darr = mdtbl['col3'].reshape(number_of_unique_phase, number_of_unique_wavelength)

print(f"Table length: {len(_mdtbl)} --> {len(mdtbl)}")


param_keys = infotbl.keys()[2:]
param_keys
# %% [markdown]
# ## Input Tables
# for inexptime in [60, 180, 300, 600, 900]:
# for inexptime in [900]:
for inexptime in [60, 180, 300, 600, 900]:


	# %%
	if group == 'med25nm':
		#	Medium-band
		path_input = f'../input/{intype}/{indist:0>3}Mpc/{inexptime:0>3}s/{group}'
	else:
		#	Broad-band
		path_input = f'../input/{intype}/{indist:0>3}Mpc/{inexptime:0>3}s/broad'
	path_output = f'../fit_result/{intype}2{fittype}/{indist:0>3}Mpc/{inexptime:0>3}s/{group}'
	if not os.path.exists(path_output):
		os.makedirs(path_output)
	outfits = f"{path_output}/fit_result.fits"

	# %%
	intablelist = sorted(glob.glob(f"{path_input}/obs.*.fits"))
	print(f"{len(intablelist)} input tables found")

	# %%
	outbl = Table()
	#	Input data
	outbl['input_table'] = [os.path.basename(intable) for intable in intablelist]
	#	Detection / Fit
	outbl['ndet'] = 0
	outbl['det_filters'] = " "*200
	outbl['det'] = False
	outbl['fit'] = False
	#	Fitted Parameters
	outbl['z'] = 0.
	outbl['t'] = 0.
	for key in param_keys:
		outbl[key] = 0.
	#	Error
	outbl['zerr'] = 0.
	outbl['terr'] = 0.
	for key in param_keys:
		outbl[f"{key}err"] = 0.

	#	Fit Results
	outbl['free_params'] = 0
	outbl['dof'] = 0
	outbl['chisq'] = 0.
	outbl['chisqdof'] = 0.
	outbl['bic'] = 0.
	#	Meta
	outbl.meta['fittype'] = fittype
	outbl.meta['source'] = source
	outbl.meta['intype'] = intype
	outbl.meta['indist[Mpc]'] = indist
	outbl.meta['inexptime[s]'] = inexptime
	outbl.meta['group'] = group
	outbl.meta['date'] = Time.now().isot

	# %%
	#	Temp Table
	_mdtbl = Table.read(intablelist[0])
	for key, val in _mdtbl.meta.items():
		if key in ['MD', 'VD', 'MW', 'VW', 'ANG', 'PHASE', 'REDSHIFT',]:
			if type(val) is str:
				outbl[key] = ' '*10
			elif type(val) is float:
				outbl[key] = 0.0
			elif type(val) is int:
				outbl[key] = 0
			# print(key, val)


	ii = 10
	intable = intablelist[ii]
	st = time.time()
	# %%
	for ii, intable in enumerate(intablelist):
		#%%
		print(f"{os.path.basename(intable)} ({inexptime}s) --> {fittype}")
		intbl = Table.read(intable)

		# %% [markdown]
		# - Detection

		# %%
		indx_det = np.where(intbl['snr']>snrcut)
		filterlist_det = list(intbl['filter'][indx_det])
		filterlist_str = ",".join(filterlist_det)

		# %%
		filterset = [f"{group}-{filte}" for filte, _, group in filterset_group if filte in filterlist_det]
		bands = speclite.filters.load_filters(*filterset)

		# %%
		ndet = len(filterset)
		detection = np.any(intbl['snr'] > 5)
		if verbose:
			print(f"number of detections: {ndet}")
			print(f"detection: {detection}")


		# %%
		if detection:
			xdata = intbl['fnuobs'].data[indx_det]
			ydata = xdata
			sigma = intbl['fnuerr'].data[indx_det]

			p0 = (
				0.01, np.median(phasearr), np.median(infotbl['pc1']), np.median(infotbl['pc2']), np.median(infotbl['pc3'])
			)

			bounds = (
				(np.min(zarr), np.min(phasearr), np.min(infotbl['pc1']), np.min(infotbl['pc2']), np.min(infotbl['pc3'])),
				(np.max(zarr), np.max(phasearr), np.max(infotbl['pc1']), np.max(infotbl['pc2']), np.max(infotbl['pc3']))
			)

			# %%
			n_free_param = len(inspect.signature(func).parameters)-1

			# %%
			#	Default `fit`
			fit = False
			try:
				popt, pcov = curve_fit(
					func,
					xdata=xdata,
					ydata=ydata,
					sigma=sigma,
					p0=p0,
					absolute_sigma=True,
					check_finite=True,
					bounds=bounds,
					method='trf',
					# max_nfev=1e4,
				)
				fit = True
			except Exception as e:
				# print(e)
				outlog = f"{path_output}/{os.path.basename(intable).replace('obs', 'fit').replace('fits', 'log')}"
				f = open(outlog, 'w')
				f.write(str(e))
				f.close()
				fit = False

			# %%
			if fit:
				#	Fitting result
				r = ydata.data - func(xdata, *popt)
				n_free_param = len(inspect.signature(func).parameters)-1
				dof = ndet - n_free_param
				chisq_i = (r / sigma) ** 2
				chisq = np.sum(chisq_i)
				chisqdof = chisq/dof
				bic = chisq + n_free_param*np.log(ndet)
				perr = np.sqrt(np.diag(pcov))

				# %%
				z, t, pc1, pc2, pc3 = popt

				if verbose:
					for param, val in zip(param_keys, popt):
						print(f"{param}: {val:.3f}")
				# %%
				outpng = f"{path_output}/{os.path.basename(intable).replace('obs', 'fit').replace('fits', 'png')}"


				# %%
				parameters = ['z', 't']+param_keys
				#	Detection / Fit
				outbl['ndet'][ii] = ndet
				outbl['det_filters'][ii] = filterlist_str
				outbl['det'][ii] = detection
				outbl['fit'][ii] = fit
				#	Fitted Parameters
				for key, val, valerr in zip(parameters, popt, perr):
					#	Value
					outbl[key][ii] = val
					#	Error
					outbl[f"{key}err"][ii] = valerr
				#	Fit Results
				outbl['free_params'][ii] = n_free_param
				outbl['dof'][ii] = dof
				outbl['chisq'][ii] = chisq
				outbl['chisqdof'][ii] = chisqdof
				outbl['bic'][ii] = bic


				# %%
				#	Temp Table
				for key, val in intbl.meta.items():
					if key in ['MD', 'VD', 'MW', 'VW', 'ANG', 'PHASE', 'REDSHIFT',]:
						outbl[key][ii] = val

				# %% [markdown]
				# # Plot

				# %%
				z0 = 0

				point = (
					pc1,
					pc2,
					pc3,
					t,
					_lamarr,
				)

				iflux = interp(point)
				(zspappflam, zsplam) = apply_redshift_on_spectrum(iflux*flamunit, _lamarr*lamunit, z, z0)
				mags = bands.get_ab_magnitudes(zspappflam, zsplam)

				spmag = np.array([mags[key][0] for key in mags.keys()])
				spfnu = (spmag*u.ABmag).to(u.uJy).value

				# %%
				fnuarr = convert_flam2fnu(zspappflam, zsplam).to(u.uJy)

				# %%
				label = f"""n_det={ndet}, rchisq={chisqdof:.3}, bic={bic:.3}
				"""
				for kk, (key, val, valerr) in enumerate(zip(parameters, popt, perr)):
					if (kk == 6) or (kk == 12):
						label += '\n'
					label += f"{key}={val:.3}, "
				# %%
				plt.close('all')
				plt.figure(figsize=(8, 6))
				plt.plot(_lamarr, fnuarr, c='grey', lw=3, alpha=0.5, label=label)
				yl, yu = plt.ylim()
				# plt.scatter(bands.effective_wavelengths, xdata, c=intbl['snr'], marker='s', s=50, ec='k')
				plt.scatter(intbl['lam'], intbl['fnuobs'], c=intbl['snr'], marker='s', s=50, ec='k')
				if 'med' in group:
					plt.errorbar(intbl['lam'], intbl['fnuobs'], xerr=bandwidth/2, yerr=intbl['fnuerr'], c='k', ls='none', zorder=0)
				elif 'broad' in group:
					# plt.errorbar(intbl['lam'], intbl['fnuobs'], xerr=bandwidtharr_broad[indx_det]/2, yerr=intbl['fnuerr'], c='k', ls='none', zorder=0)					
					plt.errorbar(intbl['lam'], intbl['fnuobs'], xerr=bandwidtharr_broad/2, yerr=intbl['fnuerr'], c='k', ls='none', zorder=0)					
				cbar = plt.colorbar()
				cbar.set_label("SNR")
				# plt.plot(bands.effective_wavelengths, func(xdata, *popt), '.', c='tomato')
				plt.plot(intbl['lam'], intbl['fnu'], c='tomato', marker='.', ls='none', zorder=0)
				plt.title(f"{fittype} {source.upper()}")
				plt.xticks(fontsize=12)
				plt.yticks(fontsize=12)
				if 'med' in group:
					xl = 3750
					xr = 9000
				else:
					xl = 3000
					xr = 10000
				plt.xlim([xl, xr])
				plt.ylim([yl, yu])
				plt.xlabel(r"$\rm \lambda$ [$\AA$]")
				plt.ylabel(r"$\rm f_\nu$ [uJy]")
				plt.legend(loc='lower center')
				plt.tight_layout()
				plt.savefig(outpng, dpi=100)

# %%
ed = time.time()
delt = ed-st
outbl.meta['time[s]'] = delt
outbl.write(outfits, overwrite=True)
print(f"Done ({delt:.3}s)")