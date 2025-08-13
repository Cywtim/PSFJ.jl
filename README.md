# PSFJ

This a Julia package for PSF(Point spread function) inspired by psfr-python.
See more usage in <a href="test/Tutorial.ipynb">Tutorial.ipynb</a>


## Load the packages
```
using PSFJ
using PSFJ.PsfUtil
using PSFJ.KernelUtil
```

## JWST star
We using the JWFS star image as an example of the package
```
star_list = PSFJ.PsfUtil.JwstExampleStars();
```
<p align="center">
<img src="images/star.png" alt="result" width="70%">
</p>

## PSF reconstruction
The result of the psf function reconstruction

### stacking psf
Setting the kwargs of PSF:
```
# Define the kwargs of PSF
kwargs_psf_stacking = Dict(:stacking_option=>"mean")
# Stacking method, optional {"mean",  "median" or "median_weight"}
kwargs_one_step=Dict(:verbose=>false, 
                 :oversampled_residual_deshifting=>true,
                 :step_factor=>0.5,
                 :deshift_order=>1);
```
Then passing the parameters to `StackPsf`:
```
psf_guess, center_list_psfr, mask_list, amplitude_list= PSFJ.StackPsf(star_list; oversampling=1, 
    saturation_limit=nothing, num_iteration=20, 
    n_recenter=5, kwargs_psf_stacking=kwargs_psf_stacking,
    kwargs_one_step=kwargs_one_step);
```
The result in log-scale:
<p align="center">
<img src="images/psf.png" alt="result" width="70%">
</p>

### oversampling
Oversampling dict:
```
kwargs_psf_stacking = Dict(:stacking_option=>"mean")
# Stacking method, optional {"mean",  "median" or "median_weight"}

kwargs_one_step=Dict(:verbose=>false, 
                 :oversampled_residual_deshifting=>true,
                 :step_factor=>0.5,
                 :deshift_order=>1);
```
The image of oversampled psf map via
```
psf_psfr_super, center_list_psfr_super, mask_list, amplitude_list_super = PSFJ.StackPsf(star_list; oversampling=4, 
                                                                        saturation_limit=nothing, num_iteration=15, 
                                                                        n_recenter=5, kwargs_psf_stacking=kwargs_psf_stacking,
                                                                        kwargs_one_step=kwargs_one_step);
```
<p align="center">
<img src="images/psf_oversampling.png" alt="result" width="70%">
</p>

### degraded psf
Setting the Degrade Kernel
```
psf_psfr_super_degraded = KernelUtil.DegradeKernel(psf_psfr_super, 4)
psf_psfr_super_degraded = KernelUtil.CutPsf(psf_psfr_super_degraded, 91);
```
The result of degraded the oversampling psf
<p align="center">
<img src="images/psf_degraded.png" alt="result" width="70%">
</p>


## Errormap
Once the degbraded psf is solved, the errormap of point source is
```
error_map = PSFJ.PsfErrorMap(star_list, psf_psfr_super_degraded, center_list_psfr_super);
```
<p align="center">
<img src="images/errormap.png" alt="result" width="70%">
</p>




[![Build Status](https://github.com/Cywtim/PSFJ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Cywtim/PSFJ.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/Cywtim/PSFJ.jl.svg?branch=main)](https://app.travis-ci.com/Cywtim/PSFJ.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/Cywtim/PSFJ.jl?svg=true)](https://ci.appveyor.com/project/Cywtim/PSFJ-jl)
[![Coverage](https://codecov.io/gh/Cywtim/PSFJ.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Cywtim/PSFJ.jl)
[![Coverage](https://coveralls.io/repos/github/Cywtim/PSFJ.jl/badge.svg?branch=main)](https://coveralls.io/github/Cywtim/PSFJ.jl?branch=main)
