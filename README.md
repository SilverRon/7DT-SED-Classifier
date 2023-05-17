# 7DT-SED-Classifier
The 7DT telescope in the 20s will be used with 20-40 medium-band filters to obtain the SED of transients and classify them to identify kilonovae. A pipeline identifies and classifies transients based on their SEDs.

# Transient to Classify
- KN (Wollaeger+19): Use interpolator

## Structured Data (Use Interpolator)
- SN Ia (`sncosmo`: `salt3`)
	- ['z', 't0', 'x0', 'x1', 'c']
- SN Ia-91bg (`sncosmo`: `nugent-sn91bg`): Use interpolator
	- ['z', 't0', 'amplitude'] (`sncosmo`) --> use this!
	- ['model', 'stretch', 'color'] (`PlasTiCC`)
- SN Iax (`PlasTiCC`: `MOSFiT`)
	- ['model', 'Iax_INDEX', 'M_V', 't_rise', 'dm15B', 'dm15R']
	- Read model/PLAsTiCC/SNIax/sn2005hk.source.pickle with `sncosmo`
		- ['z', 't0', 'amplitude']
	- M_V: -19~-15

## Unstructured Data (Use Template fitting)
- SN Ibc (`PlasTiCC`: `MOSFiT`)
	- ['model', 'IBC_INDEX', 'Mejecta', 'Kinetic_energy', 'F_nickel']
- SN II (`PlasTiCC`: `MOSFiT`)
- SLSN-I (`PlasTiCC`: `MOSFiT`)
	- ['model', 'Iax_INDEX', 'M_V', 't_rise', 'dm15B', 'dm15R']
- TDE (`PlasTiCC`: `MOSFiT`)
	- ['model', 'TDE_INDEX', 'Rph0', 'Tvis', 'b', 'bhmass', 'eff', 'lphoto', 'starmass']

## Only Light Curve Provided
- M dwarf flare (`PlasTiCC`)
- AGN (`PlasTiCC`)

# Code
- src/GenerateSimObsData4KN.ipynb: Generate simulated observation data for KN at certain distance
- src/GenerateSimObsData4KN.py: Generate simulated observation data for KN at certain distance (python script version for execute on prompt)

## Input data
- KN observation data
	- exptime: [60, 180, 300, 900]
	- md: [0.001, 0.003, 0.01, 0.03, 0.1]
	- vd: [0.05, 0.15, 0.3]
	- mw: [0.001, 0.003, 0.01, 0.03, 0.1]
	- vw: [0.05, 0.15, 0.3]
	- angle: [0, 30, 60, 90]
