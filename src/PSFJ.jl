module PSFJ

    using  SciPy, Optim, ShiftedArrays, Statistics
    include("KernelUtil.jl")
    include("PsfUtil.jl")
    include("MaskUtil.jl")

    export psfj_path, PsfErrorMap, OneStepPsfEstimate, StackPsf

    #new type

    # Export the path of this file
    psfj_path = pwd()

    function log10_scale(arr::AbstractArray)

        log10arr = zeros(size(arr))
        log10arr[ arr.>0 ] .= log10.(arr[ arr.>0 ])
        return log10arr
    end

    """
    ...
    computes linear least square amplitude to minimize
    min[(data - amp * model)^2 / variance]

    Parameters
    ----------
    data : 2d array
        the measured data (i.e. of a star or other point-like source)
    model: 2d array, same size as data
        model prediction of the data, modulo a linear amplitude scaling
        (i.e. a PSF model that is sub-pixel shifted to match the astrometric position of the star in the data)
    variance : None, or 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels
    mask : None, or 2d integer or boolean array, same size as data
        Masking pixels not to be considered in fitting the linear amplitude;
        zeros are ignored, ones are taken into account. None as input takes all pixels into account (default)

    Returns
    -------
    amp : float
        linear amplitude such that (data - amp * model)^2 / variance is minimized
    ...
    """
    function _LinearAmplitude(data::AbstractArray, model::AbstractArray; 
        variance=nothing,
         mask=nothing)

        y = KernelUtil.Image2Array(data)
        x = KernelUtil.Image2Array(model)

        if mask !== nothing
            mask_ = KernelUtil.Image2Array(mask)
            x = x[findall(mask_ .== 1)]
            y = y[findall(mask_ .== 1)]
        else
            mask_ = nothing
        end

        if variance === nothing
            w = 1  # we simply give equal weight to all data points
        else
            w = KernelUtil.Image2Array(1. ./ variance)
            if mask_ !== nothing  # mask is not None:
                w = w[findall(mask_ .== 1)]
            end
        end
        wsum = sum(w)
        xw = sum(w .* x) ./ wsum
        yw = sum(w .* y) ./ wsum
        amp = sum(w .* (x .- xw) .* (y .- yw)) ./ sum(w .* (x .- xw) .^ 2)
        return amp
    end

    """
    ...
    shift PSF to the star position. Optionally degrades to the image resolution afterwards

    Parameters
    ----------
    psf_center : 2d array with odd square length
        Centered PSF in the oversampling space of the input
    oversampling : integer >= 1
        oversampling factor per axis of the psf_center relative to the data and coordinate shift
    shift : [x, y], 2d floats
        off-center shift in the units of the data
    degrade : boolean
        if True degrades the shifted PSF to the data resolution and cuts the resulting size to n_pix_star
    n_pix_star : odd integer
        size per axis of the data, used when degrading the shifted {SF
    order : integer >=0
        polynomial order of the ndimage.shift interpolation

    Returns
    -------
    psf_shifted : 2d array, odd axis number
        shifted PSF, optionally degraded to the data resolution
    ...
    """
    function ShiftPsf(psf_center; shift, oversampling=1, degrade=true, n_pix_star=1.,deshift_order=1)
        shift_x = shift[1] .* oversampling
        shift_y = shift[2] .* oversampling
        # shift psf
        # todo: what is optimal interpolation in the shift, or doing it in Fourier space instead?
        # Partial answer: interpolation in order=1 is better than order=0 (order=2 is even better for Gaussian PSF's)
        # The reason for (shift_x, shift_y) is the different alignment of rows and columns between julia and python
        psf_shifted = SciPy.ndimage.shift(psf_center, (shift_x, shift_y), order=deshift_order)

        # resize to pixel scale (making sure the grid definition with the center in the central pixel is preserved)
        if degrade == true
            psf_shifted = PsfUtil.Oversam2Reg(psf_shifted; oversampling=oversampling)
            psf_shifted = KernelUtil.CutEdges(psf_shifted, n_pix_star)
            # psf_shifted = CutPst(psf_shifted, n_pix_star)
        end
        return psf_shifted

    end


    """
    ...
    computes luminosity center and shifts star such that the luminosity is centered at the central pixel

    Parameters
    ----------
    star : 2d array with square odd number
        cutout of a star
    Returns
    -------
    star_shift : luminosity centered star
    ...
    """
    function LuminosityCentring(star)

        x_grid, y_grid = KernelUtil.MakeGrid(size(star)[1]; deltapix=1, left_lower=false)
        x_grid, y_grid = KernelUtil.Array2Image(x_grid), KernelUtil.Array2Image(y_grid)
        x_c, y_c =  sum(star .* x_grid) ./ sum(star), sum(star .* y_grid) ./ sum(star)
        # c_ = (len(star) - 1) / 2
        # x_s, y_s = 2 * c_ - y_c, 2 * c_ - x_c
        star_shift = ShiftPsf(star, oversampling=1, shift=[-x_c, -y_c], degrade=false, n_pix_star=size(star)[1])
        
        return star_shift
    end


    """
    ...
    Basic stacking of stars in luminosity-weighted and mask-excluded regime.
    The method ignores sub-pixel off-centering of individual stars nor does it provide an oversampled solution.
    This method is intended as a baseline comparison and as an initial guess version for the full PSF-r features.

    Parameters
    ----------
    star_list : array of 2d arrays
        list of cutout stars (to be included in the fitting)
    mask_list : array of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    symmetry: integer >= 1
        imposed symmetry of PSF estimate

    Returns
    -------
    star_stack_base : 2d array of size of the stars in star_list
        luminosity weighted and mask-excluded stacked stars
    ...
    """
    function BaseStacking(star_list, mask_list; symmetry=1)
        star_stack_base = zeros(size(star_list[1]))
        weight_map = zeros(size(star_list[1]))
        angle = 2 .* pi ./ symmetry
        for (i, star) in enumerate(star_list)
            for k in 1:symmetry
                star_shift = LuminosityCentring(star)
                star_rotated = KernelUtil.RotateImage(star_shift, angle .* (k-1))
                mask_rotated = KernelUtil.RotateImage(mask_list[i], angle .* (k-1))
                star_stack_base = star_stack_base .+ star_rotated .* mask_rotated
                weight_map = weight_map .+ mask_rotated .* sum(star)
            end
        end
        # code can't handle situations where there is never a non-zero pixel
        weight_map[ weight_map .== 0.0 ] .= 1e-12 


        star_stack_base = star_stack_base ./ weight_map

        return star_stack_base

    end


    """
    ...
    updates psf estimate based on old kernel and several new estimates

    Parameters
    ----------
    kernel_list_new : array of 2d arrays
        new PSF kernels estimated from the point sources in the image (un-normalized)
    kernel_old : 2d array of shape of the oversampled kernel
        old PSF kernel
    mask_list : None or array of booleans of shape of kernel_list_new
        masks used in the 'kernel_list_new' determination. These regions will not be considered in the combined PSF.
    amplitude_list : None or list of floats with positive semi-definite values
        pre-normalized amplitude of the different new kernel estimates (i.e. brightness of the stars, etc)
    factor : weight of updated estimate based on new and old estimate, factor=1 means new estimate,
        factor=0 means old estimate
    stacking_option : string
        option of stacking, mean or median
    symmetry: integer >= 1
        imposed symmetry of PSF estimate
    combine_with_old : boolean
        if True, adds the previous PSF as one of the proposals for the new PSFs
    error_map_list : None, or list of 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.

    Returns
    -------
    kernel_return : updated PSF estimate and error_map associated with it
    ...
    """
    function CombinePsf(kernel_list_new, kernel_old;
         mask_list=nothing,
             amplitude_list=nothing,
              factor=1., stacking_option="median",
                symmetry=1., combine_with_old=false,
                 error_map_list=nothing)

        n = Int(length(kernel_list_new) .* symmetry)

        if amplitude_list === nothing
            amplitude_list = ones(length(kernel_list_new))
        end
        if error_map_list === nothing
            error_map_list = [ ones(size(kernel_old)) for i in  1:length(kernel_list_new) ]
        end

        angle = 2*pi / symmetry # KernelUtil.RotateImage use arc unit 
        kernelsize = size(kernel_old)[1]
        kernel_list = [ zeros((kernelsize, kernelsize)) for i in 1:n ]
        weights = [ zeros(kernelsize, kernelsize) for i in 1:n ]
        i = 1 # julia array start with 1
        for (j, kernel_new) in enumerate(kernel_list_new)
            if mask_list === nothing
                mask = ones(Int, size(kernel_new))
            else
                mask = mask_list[j]
            end

            error_map_list[j][findall(error_map_list[j] .< 1e-10)] .= 10 .^ (-10)
            for k in 1:symmetry
                kernel_rotated = KernelUtil.RotateImage(kernel_new, angle .* k)
                error_map = KernelUtil.RotateImage(error_map_list[j], angle .* k)
                mask_rot = KernelUtil.RotateImage(mask, angle .* k)
                kernel_norm = KernelUtil.KernelNorm(kernel_rotated)
                kernel_list[i] .= kernel_norm
                # weight according to surface brightness, inverse variance map, and mask
                weights[i] .= amplitude_list[j] .* 1 ./ error_map .* mask_rot
                i = i + 1
            end
        end

        if combine_with_old == true
            kernel_old_rotated = [zeros((kernelsize, kernelsize)) for i in 1:symmetry]
            for i in 1:symmetry
                kernel_old_rotated[i] .= kernel_old ./ sum(kernel_old)
                kernel_list = push!(kernel_list, kernel_old_rotated)
                # todo: this next line is ambiguous about what weight being used for the old kernel estimate
                weights = push!(weights, kernel_old)
            end
        end

        # todo: outlier detection?
        if stacking_option == "median_weight"
            # todo: this is rather slow as it needs to loop through all the pixels
            # adapted from this thread:
            # https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
            flattened_psfs = [vcat(y...) for y in kernel_list]
            flattened_weights = [vcat(y...) for y in weights]
            x_dim, y_dim = size(kernel_list[1])
            new_img = []
            for i in 1:x_dim*y_dim
                pixels = flattened_psfs[:, i]
                cumsum = cumsum(flattened_weights[:, i])  # weights
                cutoff = sum(flattened_weights[:, i]) ./ 2.0  # weights
                pixels = sort(pixels)
                median = pixels[findall(cumsum .>= cutoff)][1]
                push!(new_img, median)
            end
            kernel_new = reshape(new_img, (x_dim, y_dim))'

        elseif stacking_option == "mean"
            kernel_new = [kernel_list[i] .* weights[i] ./ sum(weights[i]) for i in eachindex(kernel_list)]
            kernel_new = mean(kernel_new)

        elseif stacking_option == "median"
            if mask_list === nothing
                kernel_new = median(stack(kernel_list), dims=3)[:,:,1] # functions as np.median(array, axis=0)
            else
                # ignore masked pixels instead of over-writing with old one
                kernel_new = MedianWithMask(kernel_list, mask_list)
            end
        else
            error("stack_option must be 'median', 'median_weight' or 'mean', $stacking_option is not supported.")
        end

        findall(kernel_new === missing) .= 0.0
        # kernel_new[kernel_new < 0] = 0
        kernel_new = KernelUtil.KernelNorm(kernel_new)

        kernel_return = factor .* kernel_new .+ (1. .- factor) .* kernel_old

        return kernel_return
    end


    """
    ...
    computes the excess variance in the normalized residuals.
    This quantity can be interpreted as a linearly scaled variance term proportional to the flux of the PSF
    <point source variance>(i, j) = error_map(i, j) * <integrated point source flux>

    Parameters
    ----------
    star_list:  of arrays (2D) of stars. Odd square axis shape.
        Cutout stars from images with approximately centered on the center pixel. All stars need the same cutout size.
    psf_kernel : 2d array with square odd axis normalized to sum = 1
        PSF model in the oversampled resolution prior to this iteration step
    center_list : list of 2d floats
        list of astrometric centers relative to the center pixel of the individual stars
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    oversampling : integer >=1
        higher-resolution PSF reconstruction and return
    error_map_list : None, or list of 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.

    Returns
    -------
    psf_error_map : 2d array of the size of the pixel grid
        variance in the normalized PSF such that
        <point source variance>(i, j) = error_map(i, j) * <integrated point source flux>
    ...
    """
    function PsfErrorMap(star_list, psf_kernel, center_list,
         mask_list=nothing,
          error_map_list=nothing, oversampling=1)
        # creating oversampled mask
        if mask_list === nothing
            for (i, star) in enumerate(star_list)
                mask_list_ = [ ones(size(star)) for i in eachindex(star_list)]
            end
        else
            mask_list_ = mask_list
        end

        if error_map_list === nothing
            error_map_list = [ nothing for i in eachindex(star_list) ]
        end
        norm_residual_list = [ zeros(size(star_list[1])) for i in eachindex(star_list) ]

        for (i, star) in enumerate(star_list)
            center = center_list[i]
            # shift PSF guess to estimated position of star
            # shift PSF to position pre-determined to be the center of the point source, and degrade it to the image
            psf_shifted = ShiftPsf(psf_kernel; oversampling=oversampling, shift=center, degrade=false, n_pix_star=size(star)[1])

            # make data degraded version
            psf_shifted_data = PsfUtil.Oversam2Reg(psf_shifted; oversampling=oversampling)
            # make sure size is the same as the data and normalized to sum = 1
            psf_shifted_data = KernelUtil.CutPsf(psf_shifted_data, size(star)[1])
            # linear inversion in 1d
            amp = _LinearAmplitude(star, psf_shifted_data, variance=error_map_list[i], mask=mask_list_[i])
            residuals = abs.(star .- amp .* psf_shifted_data) .* mask_list_[i]
            # subtract expected noise level
            if error_map_list[i] !== nothing
                residuals = residuals .- sqrt.(error_map_list[i])
            end
            # make sure absolute residuals are none-negative
            residuals[residuals .< 0] .= 0
            # estimate relative error per star
            residuals = residuals ./ amp
            norm_residual_list[i] = residuals .^ 2
        end
        error_map_psf = PsfUtil.MedianWithMask(norm_residual_list, mask_list_)
        # error_map_psf[psf_kernel > 0] /= psf_kernel[psf_kernel > 0] ** 2
        error_map_psf[error_map_psf .=== missing] .= 0
        error_map_psf[error_map_psf .> 1] .= 1  # cap on error to be the same
        return error_map_psf
    end


    """
    ...
    fit the centroid of the model to the image by shifting and scaling the model to best match the data
    This is done in a non-linear minimizer in the positions (x, y) and linear amplitude minimization on the fly.
    The model is interpolated to match the data. The starting point of the model is to be aligned with the image.

    Parameters
    ----------
    data : 2d array
        data of a point-like source for which a centroid position is required
    model : 2d array, odd squared length
        a centered model of the PSF in oversampled space
    mask : None, or 2d integer or boolean array, same size as data
        Masking pixels not to be considered in fitting the linear amplitude;
        zeros are ignored, ones are taken into account. None as input takes all pixels into account (default)
    variance : None, or 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels
    oversampling : integer >= 1
        oversampling factor per axis of the model relative to the data and coordinate shift
    optimizer_type: string
        'Nelder-Mead' or 'PSO'

    Returns
    -------
    center_shift : 2d array (delta_x, delta_y)
        required shift in units of pixels in the data such that the model matches best the data
    ...
    """
    function CentroidFit(data, model;
         mask=nothing, variance=nothing,
          oversampling=1, optimizer_type="Nelder-Mead")

          init = [0., 0.] # Errrrrrrrrr
          lower_bounds = (-10., -10.)
          upper_bounds = (10., 10.)
          if mask === nothing
              mask = ones(size(data))
          end
          if variance === nothing
              variance = ones(size(data))
          end

        function _minimize(x; data=data, model=model, mask=mask,
            variance=variance, oversampling=oversampling, negative=1)

            # shift model to proposed astrometric position
            model_shifted = ShiftPsf(model; oversampling=oversampling, shift=x, degrade=true, n_pix_star=size(data)[1])
            # linear amplitude
            amp = _LinearAmplitude(data, model_shifted; variance=variance, mask=mask)
    
            chi2 = negative .* sum((data .- model_shifted .* amp) .^ 2 ./ variance .* mask)
    
            return chi2
        end

        # define the target function
        __minimize(x) = _minimize(x)

        # for testing output
        #println(__minimize((0.0,0.0)))

        if optimizer_type == "Nelder-Mead"
            # ? Optim.optimize 
            result = Optim.optimize(__minimize,
                             lower_bounds, upper_bounds,
                                init,  NelderMead())
            return Optim.minimizer(result)

        elseif optimizer_type == "LBFGS"
            result = Optim.optimize(__minimize,
                                lower_bounds, upper_bounds,
                                    init,  LBFGS())
            return Optim.minimizer(result)
        else
            erroe("optimization type $optimizer_type is not supported. Please use Nelder-Mead or LBFGS")
        end

    end


    """
    ...
    Parameters
    ----------
    star_list: list of arrays (2D) of stars. Odd square axis shape.
        Cutout stars from images with approximately centered on the center pixel. All stars need the same cutout size.
    psf_guess : 2d array with square odd axis
        best guess PSF in the oversampled resolution prior to this iteration step
    center_list : list of 2d floats
        list of astrometric centers relative to the center pixel of the individual stars
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    oversampling : integer >=1
        higher-resolution PSF reconstruction and return
    error_map_list : None, or list of 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.
    oversampled_residual_deshifting : boolean
        if True; produces first an oversampled residual map and then de-shifts it back to the center for each star
        if false; produces a residual map in the data space and de-shifts it into a higher resolution residual map for
        stacking
    deshift_order : integer >= 0
        polynomial order of interpolation of the de-shifting of the residuals to the center to be interpreted as
        desired corrections for a given star
    step_factor : float or integer in (0, 1]
        weight of updated estimate based on new and old estimate;
        psf_update = step_factor * psf_new + (1 - step_factor) * psf_old
    kwargs_psf_stacking: keyword argument list of arguments going into combine_psf()
        stacking_option : option of stacking, 'mean',  'median' or 'median_weight'
        symmetry: integer, imposed symmetry of PSF estimate
    verbose : boolean
        If True, provides plots of intermediate products walking through one iteration process for each individual star
    ...
    """
    function OneStepPsfEstimate(star_list, psf_guess, center_list; mask_list=nothing, error_map_list=nothing,
             oversampling=1, step_factor=0.2, oversampled_residual_deshifting=true, deshift_order=1,
              verbose=false, kwargs_psf_stacking=Dict(:stacking_option=>"mean"))
    
        # creating oversampled mask
        if isnothing(mask_list)
            mask_list_ = [ ]
            for (i, star) in enumerate(star_list)
                push!(mask_list_, ones(size(star)))
            end
            mask_list_oversampled = nothing
        else
            mask_list_ = mask_list
            mask_list_oversampled = [ ]
            for mask in mask_list_
                mask_ = PsfUtil.Reg2Oversam(mask; oversampling=oversampling)
                push!(mask_list_oversampled, mask_)
            end
        end

        if isnothing(error_map_list)
            error_map_list = [ ones(size(star_list[1])) .* 10.0.^(-30) for i in eachindex(star_list) ] # * 10.0**(-50)
        end


        error_map_list_psf = []  # list of variances in the centroid position and super-sampled PSF estimate
        psf_list_new = []
        amplitude_list = []
        for (i, star) in enumerate(star_list)
            center = center_list[i]
            # shift PSF guess to estimated position of star
            # shift PSF to position pre-determined to be the center of the point source, and degrade it to the image
            psf_shifted = ShiftPsf(psf_guess; oversampling, shift=center_list[i],
                                    degrade=false, n_pix_star=size(star)[1])

            # make data degraded version
            psf_shifted_data = PsfUtil.Oversam2Reg(psf_shifted; oversampling=oversampling)
            # make sure size is the same as the data and normalized to sum = 1
            psf_shifted_data = KernelUtil.CutPsf(psf_shifted_data, size(star)[1])
            # linear inversion in 1d
            amp = _LinearAmplitude(star, psf_shifted_data, variance=error_map_list[i], mask=mask_list_[i])
            push!(amplitude_list, amp)

            # shift error_map_list to PSF position
            # SciPy.ndimage.shift
            error_map_shifted = SciPy.ndimage.shift(error_map_list[i], (-center[2], -center[1]),order=deshift_order)
            error_map_shifted_oversampled = PsfUtil.Reg2Oversam(error_map_shifted, oversampling=oversampling)
            push!(error_map_list_psf, error_map_shifted_oversampled)

            # compute residuals on data
            if oversampled_residual_deshifting  # directly in oversampled space
                star_super = PsfUtil.Reg2Oversam(star, oversampling=oversampling)  # todo: needs only be calculated once!
                mask_super = PsfUtil.Reg2Oversam(mask_list_[i], oversampling=oversampling)
                # attention the routine is flux conserving and need to be changed for the mask,
                # in case of interpolation we block everything that has a tenth of a mask in there
                mask_super[ mask_super .< 1. ./ oversampling .^ 2 ./ 10] .= 0
                mask_super[mask_super .>= 1. ./ oversampling .^ 2 ./ 10] .= 1
                residuals = (star_super .- amp .* psf_shifted) .* mask_super
                residuals = residuals ./ amp

                # shift residuals back on higher res grid
                # inverse shift residuals
                shift_x = center[1] .* oversampling
                shift_y = center[2] .* oversampling
                residuals_shifted = SciPy.ndimage.shift(residuals, (-shift_y, -shift_x),order=deshift_order)

            else  # in data space and then being oversampled
                residuals = (star .- amp .* psf_shifted_data) .* mask_list_[i]
                # re-normalize residuals
                residuals = residuals ./ amp  # divide by amplitude of point source
                # high-res version of residuals
                residuals = repeat(residuals, inner=(oversampling,oversampling)) ./ oversampling .^ 2
                # shift residuals back on higher res grid
                # inverse shift residuals
                shift_x = center[1] .* oversampling
                shift_y = center[2] .* oversampling

                if oversampling % 2 == 1  # for odd number super-sampling
                    residuals_shifted = SciPy.ndimage.shift(residuals, (-shift_y, -shift_x),order=deshift_order)

                else  # for even number super-sampling
                    # for even number super-sampling half a super-sampled pixel offset needs to be performed
                    # todo: move them in all four directions (not only two)
                    residuals_shifted1 = SciPy.ndimage.shift(residuals, shift=[-shift_y - 0.5, -shift_x - 0.5],
                                                    order=deshift_order)
                    # and the last column and row need to be removed
                    residuals_shifted1 = residuals_shifted1[1:end, 1:end]

                    residuals_shifted2 = SciPy.ndimage.shift(residuals, shift=[-shift_y + 0.5, -shift_x + 0.5],
                                                    order=deshift_order)
                    # and the last column and row need to be removed
                    residuals_shifted2 = residuals_shifted2[1:end, 1:end]
                    residuals_shifted = (residuals_shifted1 .+ residuals_shifted2) ./ 2
                end
            end

            # re-size shift residuals
            psf_size = size(psf_guess)[1]
            residuals_shifted = KernelUtil.CutEdges(residuals_shifted, psf_size)
            # normalize residuals
            # remove noise from corrections
            # todo: normalization correction
            # todo: make sure without error_map that no correction is performed
            correction = residuals_shifted  # - np.sign(residuals_shifted) * np.minimum(np.sqrt(error_map_shifted_oversampled)/amp, np.abs(residuals_shifted)) # - np.mean(residuals_shifted)
            psf_new = psf_guess .+ correction
            # todo: for negative pixels, apply an average correction with its neighboring pixels
            # psf_new[psf_new < 0] = 0
            # re-normalization can bias the PSF for low S/N ratios
            # psf_new /= np.sum(psf_new)
            #= fill this function in later edition
            if verbose:
                fig = verbose_util.verbose_one_step(star, psf_shifted, psf_shifted_data, residuals, residuals_shifted,
                                                    correction, psf_new)
                fig.show()
            =#
            push!(psf_list_new, psf_new)

        end
        # stack all residuals and update the psf guess
        psf_stacking_options = Dict(:stacking_option=>"median", :symmetry=>1)
        psf_stacking_options = merge(psf_stacking_options, kwargs_psf_stacking)
        amplitude_list[amplitude_list .< 0] .= 0


        # todo: None as input for mask_list if mask are irrelevant
        kernel_new = CombinePsf(psf_list_new, psf_guess; factor=step_factor, mask_list=mask_list_oversampled,
                                amplitude_list=amplitude_list, error_map_list=error_map_list_psf,
                                psf_stacking_options...)

        kernel_new = KernelUtil.CutPsf(kernel_new, size(psf_guess)[1])

        return kernel_new, amplitude_list

    end     
    
    


    """
    ...
    Parameters
    ----------
    star_list: list of arrays (2D) of stars. Odd square axis shape.
        Cutout stars from images with approximately centered on the center pixel. All stars need the same cutout size.
    oversampling : integer >=1
        higher-resolution PSF reconstruction and return
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    error_map_list : None, or list of 2d array, same size as data
        Variance in the uncorrelated uncertainties in the data for individual pixels.
        If not set, assumes equal variances for all pixels.
    saturation_limit: float or list of floats of length of star_list
        pixel values above this threshold will not be considered in the reconstruction.
    num_iteration : integer >= 1
        number of iterative corrections applied on the PSF based on previous guess
    n_recenter: integer
        Every n_recenter iterations of the updated PSF, a re-centering of the centroids are performed with the updated
        PSF guess.
    verbose : boolean
        If True, provides plots of updated PSF during the iterative process
    kwargs_one_step : keyword arguments to be passed to one_step_psf_estimate() method
        See one_step_psf_estimate() method for options
    psf_initial_guess : None or 2d array with square odd axis
        Initial guess PSF on oversampled scale. If not provided, estimates an initial guess with the stacked stars.
    kwargs_animate : keyword arguments for animation settings
        Settings to display animation of interactive process of psf reconstruction. The argument is organized as:
            {animate: bool, output_dir: str (directory to save animation in),
            duration: int (length of animation in milliseconds)}
    kwargs_psf_stacking: keyword argument list of arguments going into combine_psf()
        stacking_option : option of stacking, 'mean',  'median' or 'median_weight'
        symmetry: integer, imposed symmetry of PSF estimate
    centroid_optimizer: 'Nelder-Mead' or 'PSO'
        option for the optimizing algorithm used to find the center of each PSF in data.


    Returns
    -------
    psf_guess : 2d  array with square odd axis
        best guess PSF in the oversampled resolution
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included.
        This list is updated with all the criteria applied on the fitting and might deviate from the input mask_list.
    center_list : list of 2d floats
        list of astrometric centers relative to the center pixel of the individual stars
    ...
    """
    function StackPsf(star_list; oversampling=1, mask_list=nothing,
             error_map_list=nothing, saturation_limit=nothing, num_iteration=20,
              n_recenter=5, verbose=false, kwargs_one_step=nothing, psf_initial_guess=nothing,
               kwargs_psf_stacking=nothing, centroid_optimizer="Nelder-Mead", kwargs_animate=nothing)
        # update the mask according to settings
        mask_list, use_mask = MaskUtil.MaskConfig(star_list; mask_list=mask_list,
                                                    saturation=saturation_limit)
        if kwargs_one_step === nothing
            kwargs_one_step = Dict()
        end
        if kwargs_psf_stacking === nothing
            kwargs_psf_stacking = Dict()
        end
        if error_map_list === nothing
            error_map_list = [ ones(size(star_list[1])) .* 10.0.^(-30) for i in eachindex(star_list) ]
        end
        if kwargs_animate === nothing
            kwargs_animate = Dict()
        end

        # update default options for animations
        animation_options = Dict("animate"=>false, "output_dir"=>"stacked_psf_animation.gif", "duration"=>5000)
        animation_options = merge(animation_options, kwargs_animate)
        # define base stacking without shift offset shifts
        # stacking with mask weight
        star_stack_base = BaseStacking(star_list, mask_list, symmetry=4)
        star_stack_base[star_stack_base .< 0.0] .= 0.
        star_stack_base = star_stack_base ./ sum(star_stack_base)

        # estimate center offsets based on base stacking PSF estimate
        center_list = []
        for (i, star) in enumerate(star_list)
            x_c, y_c = CentroidFit(star, star_stack_base; mask=mask_list[i], optimizer_type=centroid_optimizer,
                                    variance=error_map_list[i])
            push!(center_list,[x_c, y_c])
        end

        if psf_initial_guess === nothing
            psf_guess = PsfUtil.Reg2Oversam(star_stack_base, oversampling=oversampling)
        else
            psf_guess = psf_initial_guess
        end

        if verbose
            f, axes = plt.subplots(1, 1, figsize=(4 * 2, 4))
            ax = axes
            ax.imshow(log10(psf_guess), origin="lower")
            ax.set_title("input first guess")
            plt.show()
        end

        # simultaneous iterative correction of PSF starting with base stacking in oversampled resolution
        images_to_animate = []

        if use_mask
            mask_list_one_step = mask_list
        else
            mask_list_one_step = nothing
        end

        amplitude_list = [] # make amplitude_list global variable
        psf_list = []
        for j in 1:num_iteration
            if j == 1
                kernel_new, amplitude_list = OneStepPsfEstimate(star_list, psf_guess, center_list;
                                            mask_list=mask_list_one_step,
                                            error_map_list=error_map_list, 
                                            oversampling=oversampling,
                                            kwargs_psf_stacking=kwargs_psf_stacking,
                                            kwargs_one_step...)
                push!(psf_list, kernel_new)
            else
                kernel_new, amplitude_list = OneStepPsfEstimate(star_list, psf_list[j-1], center_list;
                                            mask_list=mask_list_one_step,
                                            error_map_list=error_map_list, 
                                            oversampling=oversampling,
                                            kwargs_psf_stacking=kwargs_psf_stacking,
                                            kwargs_one_step...)
                push!(psf_list, kernel_new)
            end
            #println(center_list[1])
            #println(maximum(kernel_new),",", std(kernel_new))
            
            if j % n_recenter == 1 && j != 1
                center_list = []
                for (i, star) in enumerate(star_list)
                    x_c, y_c = CentroidFit(star, psf_list[j-1]; mask=mask_list[i], oversampling=oversampling,
                                            variance=error_map_list[i], optimizer_type=centroid_optimizer)
                    push!(center_list, [x_c, y_c])
                end
            end

            #=
            if animation_options["animate"]
                push!(images_to_animate, psf_guess)
            end
            if verbose
                plt.imshow(np.log(psf_guess), vmin=-5, vmax=-1)
                plt.title("iteration $j")
                plt.colorbar()
                plt.show()
            end
            =#

        end

        return psf_list[end], center_list, mask_list, amplitude_list



        # function that is called to update the image for the animation
        #=
        function _updatefig(i)
            img.set_data(log10(images_to_animate[i]))
            return [img]
        end

        if animation_options["animate"]
            global anim
            fig = plt.figure()
            img = plt.imshow(np.log10(images_to_animate[0]))
            cmap = plt.get_cmap("viridis")
            cmap.set_bad(color='k', alpha=1.)
            cmap.set_under('k')
            # animate and display iterative psf reconstuction
            anim = animation.FuncAnimation(fig, _updatefig, frames=len(images_to_animate),
                                        interval=int(animation_options["duration"] / len(images_to_animate)), blit=True)

            anim.save(animation_options["output_dir"])
            plt.close()
        end
        =#
    end


end