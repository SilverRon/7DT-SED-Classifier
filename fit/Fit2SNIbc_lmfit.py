# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from astropy.table import Table
import sys
sys.path.append('..')
from util.helper import *
from util.sdtpy import *
import inspect
import matplotlib.pyplot as plt
from astropy.time import Time
#
from astropy.io import ascii
from lmfit import Model

# %%
snrcut = 3
source = 'MOSFiT'
fittype = 'SNIbc'
verbose = True

# %%
path_model = f'../model/PLAsTiCC/{fittype}/SIMSED.{fittype}'
models = sorted(glob.glob(f'{path_model}/*.fits'))
print(f"{len(models)} models found") 

# %%
# path_sedinfo = f'../model/PLAsTiCC/{fittype}/SIMSED.{fittype}/SED.INFO'
# infotbl = tablize_sedinfo(path_sedinfo, models)
infotbl = ascii.read(f'../model/PLAsTiCC/{fittype}/sedinfo.dat')
infotbl['model'] = [f"{path_model}/{model.replace('dat', 'dat.fits')}" for model in infotbl['model']] 
# %%
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

# %%
param_keys = infotbl.keys()[2:]
param_keys

# %%
# 데이터 변환
X, y = prepare_rf_train_data(infotbl, param_keys, phasearr, number_of_unique_phase, number_of_unique_wavelength, phase_upper=np.max(phasearr), lam_lower=np.min(lamarr), lam_upper=np.max(lamarr))
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Random Forest 모델 생성 및 훈련
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# %%
# 테스트 데이터에 대한 예측 수행
y_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 평균 제곱 오차 (MSE) 계산
mse = mean_squared_error(y_test, y_pred)
print("평균 제곱 오차 (MSE):", mse)

# 평균 절대 오차 (MAE) 계산
mae = mean_absolute_error(y_test, y_pred)
print("평균 절대 오차 (MAE):", mae)

# 결정 계수 (R-squared) 계산
r2 = r2_score(y_test, y_pred)
print("결정 계수 (R-squared):", r2)

# %% [markdown]
# # Fitting the model

# %% [markdown]
# ## Input data

# %%
register_custom_filters_on_speclite('../util')

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
# lamarr = np.arange(lammin, lammax+lamstep, lamstep)
# _lamarr = np.arange(lammin, lammax+lamstep, 10)

print(f"lam: {lammin:.3f} - {lammax:.3f} AA")
print(f"lamstep: {lamstep:g} AA")
print(f"n_lam: {len(lamarr)}")
#%%
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
	outbl['fiterr'] = False
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
	outbl['aic'] = 0.
	outbl['rsq'] = 0.
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
	_intbl = Table.read(intablelist[0])
	for key, val in _intbl.meta.items():
		if key in ['MD', 'VD', 'MW', 'VW', 'ANG', 'PHASE', 'REDSHIFT',]:
			if type(val) is str:
				outbl[key] = ' '*10
			elif type(val) is float:
				outbl[key] = 0.0
			elif type(val) is int:
				outbl[key] = 0

	# %%
	def func(x, z, t, Mejecta, Kinetic_energy, F_nickel,):

		new_data = np.array(
			[[Mejecta, Kinetic_energy, F_nickel, t]]
			)

		#	Spectrum : wavelength & flux
		flam = rf.predict(new_data)[0]*8.358E-41

		#	Redshifted
		(zspappflam, zsplam) = apply_redshift_on_spectrum(flam*flamunit, lamarr*lamunit, z, z0=0)
		mags = bands.get_ab_magnitudes(*bands.pad_spectrum(zspappflam, zsplam))

		spmag = np.array([mags[key][0] for key in mags.keys()])
		spfnu = (spmag*u.ABmag).to(u.uJy).value
		# print(f"z={z:.3f}, t={t:.3f}, bfield={bfield:.3f}, mns={mns:.3f}, pspin={pspin:.3f}, kappa={kappa:.3f}, kappagamma={kappagamma:.3f}, mej={mej:.3f}, temp={temp:.3f}, vej={vej:.3f}")
		return spfnu

	# %%
	# ii = 10
	# ii = 20
	ii = 30
	intable = intablelist[ii]
	st = time.time()
	#%%
	# for ii, intable in enumerate(intablelist[:100]):
	for ii, intable in enumerate(intablelist):
		# %%
		print(f"{os.path.basename(intable)} ({inexptime}s) --> {fittype}")
		intbl = Table.read(intable)

		# %%
		indx_det = np.where(intbl['snr']>snrcut)
		filterlist_det = list(intbl['filter'][indx_det])
		filterlist_str = ",".join(filterlist_det)

		# %%
		filterset = [f"{group}-{filte}" for filte, _, group in filterset_group if filte in filterlist_det]
		bands = speclite.filters.load_filters(*filterset)

		# %%
		# %%
		ndet = len(filterset)
		detection = np.any(intbl['snr'] > 5)
		if verbose:
			print(f"number of detections: {ndet}")
			print(f"detection: {detection}")

		# %%
		xdata = intbl['fnuobs'].data[indx_det]
		ydata = xdata
		sigma = intbl['fnuerr'].data[indx_det]

		# %% [markdown]
		# - x, z, t, M_V, t_rise, dm15B, dm15R

		# %%
		if detection:

			# 피팅에 사용할 데이터 생성
			x = xdata
			y = ydata
			# lmfit 모델 생성
			model = Model(func)
			# 초기 파라미터 추정
			params = model.make_params()
			for key in params.keys():
				if key == 'z':
					init_val = 0.01
					lower_val = 0.0001
					upper_val = 1.0
				elif key == 't':
					init_val = np.median(phasearr)
					lower_val = phasearr.min()
					upper_val = phasearr.max()
				else:
					init_val = np.median(infotbl[key])
					lower_val = np.min(infotbl[key])
					upper_val = infotbl[key].max()
				# print(key, init_val, lower_val, upper_val)
				params[key].init_value = init_val
				params[key].min = lower_val
				params[key].max = upper_val
			
			n_free_param = len(inspect.signature(func).parameters)-1


			fit = False
			try:
				# 데이터 피팅
				result = model.fit(
					y,
					params,
					x=x,
					weights=1/sigma,
					calc_covar=True,
					method='cobyla',
					)


				# 파라미터 에러 계산
				if result.covar is not None:
					perr = np.diag(result.covar)
					fiterr = True
				else:
					perr = np.array([0.0]*n_free_param)
					fiterr = False
					pass
				# param_errors = np.sqrt(np.diag(result.covar))

				# 결과 출력
				# print(result.fit_report())
				# print("Chi-square:", chi_square)
				# print("Degrees of Freedom:", dof)
				# print("Reduced Chi-square:", reduced_chi_square)
				# print("Parameter Errors:", param_errors)

				# 피팅 결과 시각화
				# plt.plot(x, y, 'bo', label='data')
				# plt.errorbar(x, y, xerr=sigma, label='data')
				# plt.plot(x, result.best_fit, 'r-', label='fit')
				# plt.legend()

				fit = True
			except Exception as e:
				# print(e)
				outlog = f"{path_output}/{os.path.basename(intable).replace('obs', 'fit').replace('fits', 'log')}"
				f = open(outlog, 'w')
				f.write(str(e))
				f.close()
				fit = False



			if fit:
				#	Fitting result
				# r = ydata.data - func(xdata, *popt)
				# n_free_param = len(inspect.signature(func).parameters)-1
				# dof = ndet - n_free_param
				# chisq_i = (r / sigma) ** 2
				# chisq = np.sum(chisq_i)
				# chisqdof = chisq/dof
				# bic = chisq + n_free_param*np.log(ndet)
				# perr = np.sqrt(np.diag(pcov))
				chisq = result.summary()['chisqr']
				chisqdof = result.summary()['redchi']
				dof = result.summary()['nfree']
				bic = result.summary()['bic']
				aic = result.summary()['aic']
				rsq = result.summary()['rsquared']

				bestfit_values = [param.value for name, param in result.params.items()]
				z, t, Mejecta, Kinetic_energy, F_nickel, = bestfit_values


				parameters = list(inspect.signature(func).parameters.keys())[1:]

				if verbose:
					print(f"ndet={ndet}")
					print(f"chisq={chisq:.3}")
					print(f"chisqdof={chisqdof:.3}")
					print(f"bic={bic:.3}")
					print(f"z={z:.3}")
					print(f"t={t:.3}")
					for key, val in zip(parameters, bestfit_values):
						print(f"{key}={val:.3}")
				outpng = f"{path_output}/{os.path.basename(intable).replace('obs', 'fit').replace('fits', 'png')}"


				#%%
				#	Detection / Fit
				outbl['ndet'][ii] = ndet
				outbl['det_filters'][ii] = filterlist_str
				outbl['det'][ii] = detection
				outbl['fit'][ii] = fit
				outbl['fiterr'][ii] = fiterr
				#	Fitted Parameters
				for key, val, valerr in zip(parameters, bestfit_values, perr):
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
				outbl['aic'][ii] = aic
				outbl['rsq'][ii] = rsq


				# t = 30
				# bfield = 1
				new_data = np.array(
					[bestfit_values[2:]+[bestfit_values[1]]]
					)

				#	Spectrum : wavelength & flux
				flam = rf.predict(new_data)[0]*8.358E-41

				(zspappflam, zsplam) = apply_redshift_on_spectrum(flam*flamunit, lamarr*lamunit, z=bestfit_values[0], z0=0)
				mags = bands.get_ab_magnitudes(*bands.pad_spectrum(zspappflam, zsplam))

				spmag = np.array([mags[key][0] for key in mags.keys()])
				spfnu = (spmag*u.ABmag).to(u.uJy).value

				fnuarr = convert_flam2fnu(zspappflam, zsplam).to(u.uJy)

				label = f"""n_det={ndet}, rchisq={chisqdof:.3}, bic={bic:.3}
				"""
				for kk, (key, val) in enumerate(zip(parameters, bestfit_values)):
					if (kk == 6) or (kk == 12):
						label += '\n'
					label += f"{key}={val:.3}, "

				plt.close('all')
				plt.figure(figsize=(8, 6))
				plt.plot(lamarr, fnuarr, c='grey', lw=3, alpha=0.5, label=label)
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


