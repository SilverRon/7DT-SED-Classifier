# %% [markdown]
# # Library

# %%
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import sncosmo
import time

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

# %%
snrcut = 3
source = 'salt3'
fittype = 'SNIa'
verbose = False


# %% [markdown]
# ## Path

# %%
intype = 'kn'
indist = 40
inexptime = 180
# group = 'broad_ugriz'
group = input("""Choose the filterset (med25nm, broad_griz, broad_ugriz):""")
if group == 'med25nm':
	filterset_group = filterlist_med25nm
elif group == 'broad_griz':
	filterset_group = filterlist_griz
elif group == 'broad_ugriz':
	filterset_group = filterlist_ugriz
filterset = [f"{group}-{filte}" for filte, _, group in filterset_group]
bands = speclite.filters.load_filters(*filterset)

if 'med' in bands.names[0]:
	#    Medium-band
	bandwidth = 250 # [AA]
	lammin = 3000
	lammax = 10000
else:
	#	Broad-band
	bandwidth = 1000 # [AA]
	lammin = 2000
	lammax = 12000

lamstep = bandwidth/10


# %%
model = sncosmo.Model(source)
model.param_names
#	phase
phasemin, phasemax = model.mintime(), model.maxtime()
phasearr = np.arange(phasemin, phasemax, 1)
#	wavelength
lammin, lammax = model.minwave(), model.maxwave()
# lamarr = np.arange(lammin, lammax, 1)
lamarr = np.arange(lammin, lammax, lamstep)

# %%
def func(x, z, t, x0, x1, c):
	model.set(z=z, x0=x0, x1=x1, c=c)
	# model.set_source_peakabsmag(-19.0, 'bessellb', 'ab')
	#	Spectrum : wavelength & flux
	lam = np.arange(model.minwave(), model.maxwave(), 1)
	flam = model.flux(t, lam)

	#	Padding the spectrum
	mags = bands.get_ab_magnitudes(*bands.pad_spectrum(flam, lam))
	
	#	Magnitude from model spectrum
	spfnu = (np.array(list(mags[0]))*u.ABmag).to(u.uJy).value

	return spfnu





# lamarr = np.arange(lammin, lammax+lamstep, lamstep)
# _lamarr = np.arange(lammin, lammax+lamstep, 10)

print(f"lam: {lammin:.3f} - {lammax:.3f} AA")
print(f"lamstep: {lamstep:g} AA")
print(f"n_lam: {len(lamarr)}")

# for inexptime in [60, 300, 600, 900]:
# for inexptime in [600, 900]:
# %%
for inexptime in [60, 180, 300, 600, 900]:

	if group == 'med25nm':
		#	Medium-band
		path_input = f'../input/{intype}/{indist:0>3}Mpc/{inexptime:0>3}s/{group}'
	else:
		#	Broad-band
		path_input = f'../input/{intype}/{indist:0>3}Mpc/{inexptime:0>3}s/broad'
	path_output = f'../fit_result/{intype}2{fittype}/{indist:0>3}Mpc/{inexptime:0>3}s/{group}'
	if not os.path.exists(path_output):
		os.makedirs(path_output)

	if verbose:
		print(f"Input  path:{path_input}")
		print(f"Output path:{path_output}")

	#	Fit Result Table
	outfits = f"{path_output}/fit_result.fits"

	intablelist = sorted(glob.glob(f"{path_input}/obs.*.fits"))

	# %% [markdown]
	# ## Output Table

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
	outbl['x0'] = 0.
	outbl['x1'] = 0.
	outbl['c'] = 0.
	#	Error
	outbl['zerr'] = 0.
	outbl['terr'] = 0.
	outbl['x0err'] = 0.
	outbl['x1err'] = 0.
	outbl['cerr'] = 0.
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
		# %%
		print(f"{os.path.basename(intable)} ({inexptime}s) --> {fittype}")
		intbl = Table.read(intable)

		if group == 'broad_ugriz':
			indx_filter = np.where(
				(intbl['filter']=='u') |
				(intbl['filter']=='g') |
				(intbl['filter']=='r') |
				(intbl['filter']=='i') |
				(intbl['filter']=='z')
			)
			intbl = intbl[indx_filter]
		elif group == 'broad_griz':
			indx_filter = np.where(
				(intbl['filter']=='g') |
				(intbl['filter']=='r') |
				(intbl['filter']=='i') |
				(intbl['filter']=='z')
			)
			intbl = intbl[indx_filter]

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

		if detection:
			xdata = intbl['fnuobs'].data[indx_det]
			ydata = xdata
			sigma = intbl['fnuerr'].data[indx_det]

			p0 = (
				0.01, 0, 0, 0, 0,
			)

			bounds = (
				(0.0, -np.inf, -np.inf, -np.inf, -np.inf),
				(1.0, np.inf, np.inf, np.inf, np.inf)
			)

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
				# z, t, x0, x1, c
				z = popt[0]
				t = popt[1]
				x0 = popt[2]
				x1 = popt[3]
				c = popt[4]
				if verbose:
					print(f"z ={z:.3}")
					print(f"t ={t:.3}")
					print(f"x0={x0:.3}")
					print(f"x1={x1:.3}")
					print(f"c ={c:.3}")

				# %%
				#	Fitting result
				r = ydata.data - func(xdata, *popt)
				n_free_param = len(inspect.signature(func).parameters)-1
				# ndet = len(xdata)
				dof = ndet - n_free_param
				chisq_i = (r / sigma) ** 2
				chisq = np.sum(chisq_i)
				chisqdof = chisq/dof
				bic = chisq + n_free_param*np.log(ndet)
				perr = np.sqrt(np.diag(pcov))

				# print(chisq)
				if verbose:
					print(f"Reduced Chisq: {chisqdof:.3}")

				# %% [markdown]
				# ## Result

				# %%
				outpng = f"{path_output}/{os.path.basename(intable).replace('obs', 'fit').replace('fits', 'png')}"

				# %%
				#	Detection / Fit
				outbl['ndet'][ii] = ndet
				outbl['det_filters'][ii] = filterlist_str
				outbl['det'][ii] = detection
				outbl['fit'][ii] = fit
				#	Fitted Parameters
				outbl['z'][ii] = z
				outbl['t'][ii] = t
				outbl['x0'][ii] = x0
				outbl['x1'][ii] = x1
				outbl['c'][ii] = c
				#	Error
				outbl['zerr'][ii] = perr[0]
				outbl['terr'][ii] = perr[1]
				outbl['x0err'][ii] = perr[2]
				outbl['x1err'][ii] = perr[3]
				outbl['cerr'][ii] = perr[4]
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

				# %%
				outbl[ii]

				# %% [markdown]
				# ## Figure

				# %%
				model.set(z=z, x0=x0, x1=x1, c=c)
				lammin, lammax = model.minwave(), model.maxwave()
				lamarr = np.arange(lammin, lammax, 1)
				flamarr = model.flux(t, lamarr)
				fnuarr = convert_flam2fnu(flamarr*flamunit, lamarr*lamunit).to(u.uJy)

				label = f"""n_det={ndet}, rchisq={chisqdof:.3}, bic={bic:.3}
				z ={z:.3}, t ={t:.3}
				x0={x0:.3}, x1={x1:.3}, c ={c:.3}"""
				plt.close('all')
				plt.figure(figsize=(8, 6))
				plt.plot(lamarr, fnuarr, c='grey', lw=3, alpha=0.5, label=label)
				yl, yu = plt.ylim()
				# plt.scatter(bands.effective_wavelengths, xdata, c=intbl['snr'], marker='s', s=50, ec='k')
				plt.scatter(intbl['lam'], intbl['fnuobs'], c=intbl['snr'], marker='s', s=50, ec='k')
				if 'med' in group:
					bandwidtharr = bandwidtharr_med25nm
				elif group == 'broad_ugriz':
					bandwidtharr = bandwidtharr_broad_ugriz
				elif group == 'broad_griz':
					bandwidtharr = bandwidtharr_broad_griz
				plt.errorbar(intbl['lam'], intbl['fnuobs'], xerr=bandwidtharr/2, yerr=intbl['fnuerr'], c='k', ls='none', zorder=0)
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
