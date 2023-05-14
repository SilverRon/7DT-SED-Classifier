# %% [markdown]
# # Generate Simulated Observation Data for KN
# - Input data for SED classifier
# - Author: Gregory S.H. Paek (23.5.11)

# %%
from astropy.time import Time

# %% [markdown]
# # Setting

# %%
import pickle

with open('../model/gw170817-like.interp.pkl', 'rb') as f:
	interp = pickle.load(f)

# %% [markdown]
# - Kilonova Parameters

# %%
import numpy as np

#	Ejecta mass, velocity
mdarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
vdarr = np.array([0.05, 0.15, 0.3])
mwarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
vwarr = np.array([0.05, 0.15, 0.3])
#	Viewing angle
angarr = np.linspace(0, 180, 54)

#	min-max (1002.35-127,695.0 AA)
lamarr = np.arange(1003, 127695.+1, 1)

# %% [markdown]
# # Single Case

# %%
point = (
	mdarr[0],
	vdarr[0],
	mwarr[0],
	vwarr[0],
	angarr[0],
	1,
	lamarr,
)

iflux = interp(point)

# %%
import matplotlib.pyplot as plt
plt.plot(lamarr, iflux)
plt.xlim(3750, 9000)

# %%
import sys
sys.path.append('..')
from util.helper import *
from util.sdtpy import *

# %%
register_custom_filters_on_speclite('../util')

# %%
filterset = [f"{group}-{filte}" for filte, _, group in filterlist_med25nm]
bands = speclite.filters.load_filters(*filterset)

# %%
mags = bands.get_ab_magnitudes(*bands.pad_spectrum(iflux, lamarr)).to_pandas().to_numpy()[0]

# %%
fnuarr = (mags*u.ABmag).to(u.uJy)
flamarr = convert_fnu2flam(fnuarr, bands.effective_wavelengths)

# %%
plt.plot(lamarr, iflux)
plt.plot(bands.effective_wavelengths, flamarr, 'o')
plt.xlim(3750, 9000)

# %% [markdown]
# # Iteration for generating the simulated observation data

# %%
# 필요한 라이브러리 불러오기
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value

# 거리 정의 (Mpc)
distances = [40, 156] * u.Mpc

# 각 거리에 대한 적색 이동 계산
redshifts = [z_at_value(cosmo.luminosity_distance, d) for d in distances]

# 결과 출력
for i, d in enumerate(distances):
    print(f"The redshift for a distance of {d} is {redshifts[i]:.3f}")

d = 40*1e6 # [pc]
redshift = redshifts[0]


# %%
typ = 'kn'
exptime = 180
distance_Mpc = int(d/1e6) # [Mpc]
group = 'med25nm'

path_save = f'../input/{typ}/{distance_Mpc:0>3}Mpc/{exptime:0>3}s/{group}'

if not os.path.exists(path_save):
    os.makedirs(path_save)

# %% [markdown]
# ## Declare the 7DT Class

# %%
#	Subsequent filter info [AA]
bandmin=4000
bandmax=9000
bandwidth=250
bandstep=250
#	Maximum transmission of each filters
bandrsp=0.95
#	Wavelength bin [AA]
lammin=1000
lammax=10000
lamres=1000
#	Exposure Time [s]
# exptime = 60
exptime = 180
# exptime = 300

# %% [markdown]
# - medium band (bandwidth=25nm)

# %%
sdt = SevenDT()
sdt.echo_optics()
filterset = sdt.generate_filterset(bandmin=bandmin, bandmax=bandmax, bandwidth=bandwidth, bandstep=bandstep, bandrsp=bandrsp, lammin=lammin, lammax=lammax, lamres=lamres)
# sdt.plot_filterset()
T_qe = sdt.get_CMOS_IMX455_QE()
# T_comp = sdt.get_CCD_Hamamtsu_QE()
# sdt.plot_QE()
sdt.get_optics()
s = sdt.get_sky()
# sdt.plot_sky()
sdt.smooth_sky()
# sdt.plot_sky_smooth()
# sdt.plot_sky_sb()
totrsptbl = sdt.calculate_response()
Npix_ptsrc, Narcsec_ptsrc = sdt.get_phot_aperture(exptime=exptime, fwhm_seeing=1.5, optfactor=0.6731, verbose=False)
outbl = sdt.get_depth_table(Nsigma=5)
# sdt.plot_point_source_depth()
sdt.get_speclite()

# %% [markdown]
# - broad-band

# %%
sdt_broad = SevenDT()
sdt_broad.echo_optics()
filterset_broad = sdt_broad.get_sdss_filterset()
# sdt_broad.plot_filterset()
T_qe = sdt_broad.get_CMOS_IMX455_QE()
# T_comp = sdt_broad.get_CCD_Hamamtsu_QE()
# sdt_broad.plot_QE()
sdt_broad.get_optics()
s = sdt_broad.get_sky()
# sdt_broad.plot_sky()
sdt_broad.smooth_sky()
# sdt_broad.plot_sky_smooth()
# sdt_broad.plot_sky_sb()
totrsptbl = sdt_broad.calculate_response()
Npix_ptsrc, Narcsec_ptsrc = sdt_broad.get_phot_aperture(exptime=exptime, fwhm_seeing=1.5, optfactor=0.6731, verbose=False)
outbl = sdt_broad.get_depth_table(Nsigma=5)
# sdt_broad.plot_point_source_depth()
sdt_broad.get_speclite()

# %% [markdown]
# ## Iteration

# %%
phasearr = np.array([0.125, 0.25, 0.5, 1.0])
angarr = np.array([0, 30, 60, 90])
total_number = len(mdarr) * len(vdarr) * len(mwarr) * len(vwarr) * len(angarr) * len(phasearr)
print("Number of parameters:", len(mdarr), len(vdarr), len(mwarr), len(vwarr), len(angarr), len(phasearr))
print("Total Number of parameters:", total_number)

inexptime = float(input('Input exposure time [s]:'))

# exptimearr = np.array([60, 180, 300, 900])
exptimearr = np.array([inexptime,])


for exptime in exptimearr:

    typ = 'kn'
    # exptime = exptime
    distance_Mpc = int(d/1e6) # [Mpc]
    group = 'med25nm'

    path_save = f'../input/{typ}/{distance_Mpc:0>3}Mpc/{exptime:0>3}s/{group}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)


    # %%
    # len(phasearr), len(mdarr), len(vdarr), len(mwarr), len(vwarr), len(angarr)
    import itertools
    for cc, combination in enumerate(itertools.product(mdarr, vdarr, mwarr, vwarr, angarr, phasearr)):
        print(f"[{cc+1:0>4}/{total_number}] {typ} {group} {exptime}s", end='\r')
        prefix = f"obs.{cc:0>6}"
        if not os.path.exists(f"{path_save}/{prefix}.fits"):
            # print(combination)
            #
            md, vd, mw, vw, ang, phase = combination
            #
            point = (*combination, lamarr)
            #	Interpolated flux (but luminosity!)
            iflux = interp(point)
            ifnu = convert_flam2fnu(iflux*flamunit, lamarr*lamunit)
            absmag = ifnu.to(u.ABmag)-2.5*np.log10(54)*u.ABmag
            appmag = convert_abs2app(absmag.value, d=40*1e6)

            #	Distance Scaled flux
            fnu = (appmag*u.ABmag).to(u.uJy)
            flam = convert_fnu2flam(fnu, lamarr*lamunit)

            #   Medium-band
            _, __ = sdt.get_phot_aperture(exptime=exptime, fwhm_seeing=1.5, optfactor=0.6731, verbose=False)
            _ = sdt.get_depth_table(Nsigma=5)
            obstbl = sdt.get_synphot2obs(flam, lamarr*lamunit, z=None, z0=redshift, figure=True)

            #	Meta data
            obstbl.meta['md'] = md
            obstbl.meta['vd'] = vd
            obstbl.meta['mw'] = mw
            obstbl.meta['vw'] = vw
            obstbl.meta['ang'] = ang
            obstbl.meta['phase'] = phase
            obstbl.meta['type'] = typ
            obstbl.meta['group'] = group
            obstbl.meta['exptime'] = exptime
            obstbl.meta['redshift'] = redshift.value
            obstbl.meta['dist'] = d
            obstbl.meta['author'] = 'G.Paek'
            obstbl.meta['date'] = Time.now().isot
            obstbl.write(f"{path_save}/{prefix}.fits", overwrite=True)

            plt.title(f"md:{md:.3} vd:{vd:.3} mw:{mw:.3} vw:{vw:.3} ang:{ang:g} phase:{phase:.3}")
            plt.tight_layout()
            plt.savefig(f"{path_save}/{prefix}.png", dpi=100)
            plt.close('all')

        # if cc == 10: break

    print("Done!")

    # %% [markdown]
    # ## Broad-band

    # %%
    typ = 'kn'
    # exptime = 180
    distance_Mpc = int(d/1e6) # [Mpc]
    group = 'broad'

    path_save = f'../input/{typ}/{distance_Mpc:0>3}Mpc/{exptime:0>3}s/{group}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # %%
    # len(phasearr), len(mdarr), len(vdarr), len(mwarr), len(vwarr), len(angarr)
    import itertools
    for cc, combination in enumerate(itertools.product(mdarr, vdarr, mwarr, vwarr, angarr, phasearr)):
        print(f"[{cc+1:0>4}/{total_number}] {typ} {group} {exptime}s", end='\r')
        # print(combination)
        prefix = f"obs.{cc:0>6}"
        if not os.path.exists(f"{path_save}/{prefix}.fits"):
            #
            md, vd, mw, vw, ang, phase = combination
            #
            point = (*combination, lamarr)
            #	Interpolated flux (but luminosity!)
            iflux = interp(point)
            ifnu = convert_flam2fnu(iflux*flamunit, lamarr*lamunit)
            absmag = ifnu.to(u.ABmag)-2.5*np.log10(54)*u.ABmag
            appmag = convert_abs2app(absmag.value, d=40*1e6)

            #	Distance Scaled flux
            fnu = (appmag*u.ABmag).to(u.uJy)
            flam = convert_fnu2flam(fnu, lamarr*lamunit)

            #   Medium-band
            _, __ = sdt_broad.get_phot_aperture(exptime=exptime, fwhm_seeing=1.5, optfactor=0.6731, verbose=False)
            _ = sdt_broad.get_depth_table(Nsigma=5)
            obstbl = sdt_broad.get_synphot2obs(flam, lamarr*lamunit, z=None, z0=redshift, figure=True)

            #	Meta data
            obstbl.meta['md'] = md
            obstbl.meta['vd'] = vd
            obstbl.meta['mw'] = mw
            obstbl.meta['vw'] = vw
            obstbl.meta['ang'] = ang
            obstbl.meta['phase'] = phase
            obstbl.meta['type'] = typ
            obstbl.meta['group'] = group
            obstbl.meta['exptime'] = exptime
            obstbl.meta['redshift'] = redshift.value
            obstbl.meta['dist'] = d
            obstbl.meta['author'] = 'G.Paek'
            obstbl.meta['date'] = Time.now().isot
            obstbl.write(f"{path_save}/{prefix}.fits", overwrite=True)

            plt.title(f"md:{md:.3} vd:{vd:.3} mw:{mw:.3} vw:{vw:.3} ang:{ang:g} phase:{phase:.3}")
            plt.tight_layout()
            plt.savefig(f"{path_save}/{prefix}.png", dpi=100)
            plt.close('all')

        # if cc == 10: break

    print("Done!")


