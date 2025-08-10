module MaskUtil

    export MaskConfig

    NothingOrArray = Union{Nothing, Array}
    NothingOrFloat = Union{Nothing, Float64}
    NothingFloatArray = Union{Nothing, Float64, Array}

    function atleast_1d(arys::NothingFloatArray)
        res = []
        for ary in arys
            if isempty(size(ary))
                result = [ary]
            else
                result = ary
            push!(res, result)
            end
        end
        
        if len(res) == 1
            return res[1]
        else
            return res
        end
    end


    """
    ...
    configures the fitting masks for individual stars based on an optional input mask list and saturation limits
    
    Parameters
    ----------
    star_list : vector of 2d arrays
        list of cutout stars (to be included in the fitting)
    mask_list : list of 2d boolean or integer arrays, same shape as star_list
        list of masks for each individual star's pixel to be included in the fit or not.
        0 means excluded, 1 means included
    saturation: nothing, float or list of floats with same length as star_list
        saturation limit of the pixels. Pixel values at and above this value will be masked

    Returns
    -------
    mask_list : vector of 2d boolean or integer arrays, same shape as star_list
        an updated list of masks for each individual star taking into account saturation
    use_mask : boolean
        if true, indicates that the masks need to be used, otherwise calculations will ignore mask for speed-ups
    ...
    """
    function MaskConfig(star_list; mask_list::NothingOrArray=nothing, saturation::NothingFloatArray=nothing)
        if isnothing(saturation) && isnothing(mask_list)
            use_mask = false
        else
            use_mask = true
        end

        n_star = length(star_list)


        # define saturation masks
        if saturation !== nothing
            if typeof(satureation) == Float64
                satureation = [ satureation ]
            end
            n_sat = size(saturation)[1]
            if n_sat == 1
                saturation = [ saturation for i in 1:n_star]
            else
                if n_sat != n_star
                    error("saturation is a list with non-equal length to star_list.")
                end
            end
        end

        # initiate a mask_list that accepts all pixels
        if isnothing(mask_list)
            mask_list = []
            for (i, star) in enumerate(star_list)
                mask = ones(size(star))
                push!(mask_list, mask)
            end
        end
        # add threshold for saturation in mask
        if !isnothing(saturation)
            for (i, mask) in enumerate(mask_list)
                mask[findall(star_list[i] .> saturation[i])] .= 0 # need to be corrected
            end
        end

        return Matrix.(mask_list), use_mask

    end

end