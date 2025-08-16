using PSFJ
using PSFJ.PsfUtil
using PSFJ.KernelUtil
using PyPlot, BenchmarkTools, SciPy, Optim, Statistics 

star_list = PSFJ.PsfUtil.JwstExampleStars();

# Define the kwargs of PSF
kwargs_psf_stacking = Dict(:stacking_option=>"mean")
# Stacking method, optional {"mean",  "median" or "median_weight"}

kwargs_one_step=Dict(:verbose=>false, 
                 :oversampled_residual_deshifting=>true,
                 :step_factor=>0.5,
                 :deshift_order=>1);

psf_guess, center_list_psfr, mask_list, amplitude_list= PSFJ.StackPsf(star_list; oversampling=1, 
                                                  saturation_limit=nothing, num_iteration=20, 
                                                  n_recenter=5, kwargs_psf_stacking=kwargs_psf_stacking,
                                                kwargs_one_step=kwargs_one_step);