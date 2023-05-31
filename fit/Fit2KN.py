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
with open('../model/gw170817-like.interp.pkl', 'rb') as f:
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
	lammin = 2000
	lammax = 12000

# lammin = bands.effective_wavelengths.min().value-bandwidth*2
# lammax = bands.effective_wavelengths.max().value+bandwidth*2

lamstep = bandwidth/10
lamarr = np.arange(lammin, lammax+lamstep, lamstep)
_lamarr = np.arange(lammin, lammax+lamstep, 10)

print(f"lam: {lammin:.3f} - {lammax:.3f} AA")
print(f"lamstep: {lamstep:g} AA")
print(f"n_lam: {len(lamarr)}")

# %% [markdown]
# - Model parameters

# %%
mdarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
vdarr = np.array([0.05, 0.15, 0.3])
mwarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
vwarr = np.array([0.05, 0.15, 0.3])
#	Viewing angle
angarr = np.linspace(0, 180, 54)

#	Phase
phasearr = np.array([0.125, 0.25, 0.5, 1.0])

#	Redshift
zarr = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])

mdmin = np.min(mdarr)
mdmax = np.max(mdarr)
vdmin = np.min(vdarr)
vdmax = np.max(vdarr)
mwmin = np.min(mwarr)
mwmax = np.max(mwarr)
vwmin = np.min(vwarr)
vwmax = np.max(vwarr)
angmin = np.min(angarr)
angmax = np.max(angarr)
phasemin = np.min(phasearr)
phasemax = np.max(phasearr)
zmin = np.min(zarr)
zmax = np.max(zarr)

print(f"md : {np.mean(mdarr):.3} ({mdmin:.3} - {mdmax:.3})")
print(f"vd : {np.mean(vdarr):.3} ({vdmin:.3} - {vdmax:.3})")
print(f"mw : {np.mean(mwarr):.3} ({mwmin:.3} - {mwmax:.3})")
print(f"vw : {np.mean(vwarr):.3} ({vwmin:.3} - {vwmax:.3})")
print(f"ang: {np.mean(angarr):.3} ({angmin:.3} - {angmax:.3})")
print(f"phase: {np.mean(phasearr):.3} ({phasemin:.3} - {phasemax:.3})")

# %% [markdown]
# - Function to fit
# %%
def func(x, md, vd, mw, vw, ang, phase, z):

	isotropic_factor = 54.
	z0 = 0

	#	Input
	point = (
		md,
		vd,
		mw,
		vw,
		ang,
		phase,
		lamarr,
	)

	iflux = interp(point)*isotropic_factor
	(zspappflam, zsplam) = apply_redshift_on_spectrum(iflux*flamunit, lamarr*lamunit, z, z0)
	mags = bands.get_ab_magnitudes(*bands.pad_spectrum(zspappflam, zsplam))

	spmag = np.array([mags[key][0] for key in mags.keys()])
	spfnu = (spmag*u.ABmag).to(u.uJy).value

	return spfnu

# %% [markdown]
# ## Variables to set

# %%
snrcut = 3
source = 'wollaeger+21'
fittype = 'KN'
verbose = False




# for inexptime in [180, 300, 600, 900]:
# for inexptime in [60]:
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
	outbl['md'] = 0.
	outbl['vd'] = 0.
	outbl['mw'] = 0.
	outbl['vw'] = 0.
	outbl['ang'] = 0.
	#	Error
	outbl['zerr'] = 0.
	outbl['terr'] = 0.
	outbl['mderr'] = 0.
	outbl['vderr'] = 0.
	outbl['mwerr'] = 0.
	outbl['vwerr'] = 0.
	outbl['angerr'] = 0.

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

		# %% [markdown]
		# - Answer parameters

		# %%
		inmd = intbl.meta['MD']
		invd = intbl.meta['VD']
		inmw = intbl.meta['MW']
		invw = intbl.meta['VW']
		inang = intbl.meta['ANG']
		inphase = intbl.meta['PHASE']
		inz = intbl.meta['REDSHIFT']

		# %% [markdown]
		# - Normalized answer parameters

		# %%
		#	Normalized initial guess
		ninmd = (inmd - mdmin) / (mdmax - mdmin)
		ninvd = (invd - vdmin) / (vdmax - vdmin)
		ninmw = (inmw - mwmin) / (mwmax - mwmin)
		ninvw = (invw - vwmin) / (vwmax - vwmin)
		ninang = (inang - angmin) / (angmax - angmin)
		ninphase = (inphase - phasemin) / (phasemax - phasemin)
		ninz = (inz - zmin) / (zmax - zmin)
		if verbose:
			print(f"ninmd = {ninmd:.3}")
			print(f"ninvd = {ninvd:.3}")
			print(f"ninmw = {ninmw:.3}")
			print(f"ninvw = {ninvw:.3}")
			print(f"ninang = {ninang:.3}")
			print(f"ninphase = {ninphase:.3}")
			print(f"ninz = {ninz:.3}")

		# %% [markdown]
		# - Min/max value

		# %%
		mdlo = mdarr.min() if inmd * 0.9 < mdarr.min() else inmd * 0.9
		mdup = mdarr.max() if inmd * 1.1 > mdarr.max() else inmd * 1.1
		vdlo = vdarr.min() if invd * 0.9 < vdarr.min() else invd * 0.9
		vdup = vdarr.max() if invd * 1.1 > vdarr.max() else invd * 1.1
		mwlo = mwarr.min() if inmw * 0.9 < mwarr.min() else inmw * 0.9
		mwup = mwarr.max() if inmw * 1.1 > mwarr.max() else inmw * 1.1
		vwlo = vwarr.min() if invw * 0.9 < vwarr.min() else invw * 0.9
		vwup = vwarr.max() if invw * 1.1 > vwarr.max() else invw * 1.1
		# anglo = angarr.min() if inang * 0.9 < angarr.min() else inang * 0.9
		# angup = angarr.max() if inang * 1.1 > angarr.max() else inang * 1.1
		anglo = angarr.min() if inang * 0.9 < angarr.min() else inang * 0.9
		angup = angarr.max() if inang * 1.1 > angarr.max() else inang * 1.1
		if (anglo == 0) & (angup == 0):
			anglo = 0
			angup = 180
		phaselo = phasearr.min() if inphase * 0.9 < phasearr.min() else inphase * 0.9
		phaseup = phasearr.max() if inphase * 1.1 > phasearr.max() else inphase * 1.1
		if verbose:
			print(f"md: {mdlo:.3}-{mdup:.3}")
			print(f"vd: {vdlo:.3}-{vdup:.3}")
			print(f"mw: {mwlo:.3}-{mwup:.3}")
			print(f"vw: {vwlo:.3}-{vwup:.3}")
			print(f"ang: {anglo:.3}-{angup:.3}")
			print(f"phase: {phaselo:.3}-{phaseup:.3}")


		# %% [markdown]
		# - Normalized boundaries

		# %%
		# nmdlo = 0. if ninmd - 0.1 < 0. else ninmd - 0.1
		# nmdup = 1. if ninmd + 0.1 > 1. else ninmd + 0.1
		# nvdlo = 0. if ninvd - 0.1 < 0. else ninvd - 0.1
		# nvdup = 1. if ninvd + 0.1 > 1. else ninvd + 0.1
		# nmwlo = 0. if ninmw - 0.1 < 0. else ninmw - 0.1
		# nmwup = 1. if ninmw + 0.1 > 1. else ninmw + 0.1
		# nvwlo = 0. if ninvw - 0.1 < 0. else ninvw - 0.1
		# nvwup = 1. if ninvw + 0.1 > 1. else ninvw + 0.1
		# nanglo = 0. if ninang - 0.1 < 0. else ninang - 0.1
		# nangup = 1. if ninang + 0.1 > 1. else ninang + 0.1
		# nphaselo = 0. if ninphase - 0.1 < 0. else ninphase - 0.1
		# nphaseup = 1. if ninphase + 0.1 > 1. else ninphase + 0.1
		# ninzlo = 0. if ninz - 0.1 < 0. else ninz * 0
		# ninzup = 1. if ninz + 0.1 > 1. else ninz * 1

		# print(f"nmdlo: {nmdlo:.3}-{nmdup:.3}")
		# print(f"nvdlo: {nvdlo:.3}-{nvdup:.3}")
		# print(f"nmwlo: {nmwlo:.3}-{nmwup:.3}")
		# print(f"nvwlo: {nvwlo:.3}-{nvwup:.3}")
		# print(f"nanglo: {nanglo:.3}-{nangup:.3}")
		# print(f"nphaselo: {nphaselo:.3}-{nphaseup:.3}")
		# print(f"ninzlo: {ninzlo:.3}-{ninzup:.3}")

		# %% [markdown]
		# # Fitting

		# %% [markdown]
		# - Input data

		# %%
		if detection:
			xdata = intbl['fnuobs'].data[indx_det]
			ydata = xdata
			sigma = intbl['fnuerr'].data[indx_det]

			# %%
			# p0 = (
			#     #   answer
			# 	ninmd, ninvd, ninmw, ninvw, ninang, ninphase, ninz,
			#     #   mean
			# 	# 0.5, 0.5, 0.5, 0.5, 0.5, ninphase, ninz,
			# )
			# bounds = (
			#     #   answer boundary
			# 	# (0, 0, 0, 0, 0, 0, 0,),
			#     # (1, 1, 1, 1, 1, 1, 1,),
			#     (nmdlo, nvdlo, nmwlo, nvwlo, nanglo, nphaselo, ninzlo),
			#     (nmdup, nvdup, nmwup, nvwup, nangup, nphaseup, ninzup)
			# )

			# %%
			# p0 = (
			# 	inmd, invd, inmw, invw, inang, inphase, inz,
			# )
			# bounds = (
			# 	#   min-max boundary
			# 	# (mdmin, vdmin, mwmin, vwmin, angmin, phasemin, 1e-6),
			# 	# (mdmax, vdmax, mwmax, vwmax, angmax, phasemax, 1e0)
			# 	#   answer boundary
			# 	(mdlo, vdlo, mwlo, vwlo, anglo, phaselo, inz*0.9),
			# 	(mdup, vdup, mwup, vwup, angup, phaseup, inz*1.1)
			# )
			# %%
			#	Rough guess & boundaries
			# p0 = (
			# 	# inmd, invd, inmw, invw, inang, inphase, inz,
			# 	mdarr.mean(), vdarr.mean(), mwarr.mean(), vwarr.mean(), angarr.mean(), inphase, inz,
			# )
			# bounds = (
			# 	#   min-max boundary
			# 	#   answer boundary
			# 	(mdarr.min(), vdarr.min(), mwarr.min(), vwarr.min(), angarr.min(), phaselo, inz*0.9),
			# 	(mdarr.max(), vdarr.max(), mwarr.max(), vwarr.max(), angarr.max(), phaseup, inz*1.1)
			# )
			# %%
			#	Custom guess & boundaries for measuring the viewing angle
			p0 = (
				# mdarr.mean(), vdarr.mean(), mwarr.mean(), vwarr.mean(), angarr.mean(), inphase, inz,
				# inmd, invd, inmw, invw, angarr.mean(), inphase, inz,
				inmd, invd, inmw, invw, None, inphase, inz,
			)
			bounds = (
				#   min-max boundary
				#   answer boundary
				(mdlo, vdlo, mwlo, vwlo, angarr.min(), phaselo, inz*0.9),
				(mdup, vdup, mwup, vwup, angarr.max(), phaseup, inz*1.1)
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
				md = popt[0]
				vd = popt[1]
				mw = popt[2]
				vw = popt[3]
				ang = popt[4]
				phase = popt[5]
				z = popt[6]
				if verbose:
					print(f"md={md:.3}")
					print(f"vd={vd:.3}")
					print(f"mw={mw:.3}")
					print(f"vw={vw:.3}")
					print(f"ang={ang:.3}")
					print(f"phase={phase:.3}")
					print(f"z={z:.3}")

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
				outbl['t'][ii] = phase
				outbl['md'][ii] = md
				outbl['vd'][ii] = vd
				outbl['mw'][ii] = mw
				outbl['vw'][ii] = vw
				outbl['ang'][ii] = ang
				#	Error
				outbl['zerr'][ii] = perr[6]
				outbl['terr'][ii] = perr[5]
				outbl['mderr'][ii] = perr[0]
				outbl['vderr'][ii] = perr[1]
				outbl['mwerr'][ii] = perr[2]
				outbl['vwerr'][ii] = perr[3]
				outbl['angerr'][ii] = perr[4]
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
				isotropic_factor = 54.
				z0 = 0

				point = (
					md,
					vd,
					mw,
					vw,
					ang,
					phase,
					_lamarr,
				)

				iflux = interp(point)*isotropic_factor
				(zspappflam, zsplam) = apply_redshift_on_spectrum(iflux*flamunit, _lamarr*lamunit, z, z0)
				mags = bands.get_ab_magnitudes(zspappflam, zsplam)

				spmag = np.array([mags[key][0] for key in mags.keys()])
				spfnu = (spmag*u.ABmag).to(u.uJy).value

				# %%
				fnuarr = convert_flam2fnu(zspappflam, zsplam).to(u.uJy)

				# %%
				label = f"""n_det={ndet}, rchisq={chisqdof:.3}, bic={bic:.3f}
				md={md:.3f}, vd={vd:.3f}c, mw={mw:.3f}, vw={vw:.3f}c
				angle={ang:.1f}deg, phase={phase:.3f}d"""

				# %%
				plt.close('all')
				plt.figure(figsize=(8, 6))
				plt.plot(_lamarr, fnuarr, c='grey', lw=3, alpha=0.5, label=label)
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