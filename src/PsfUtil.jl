module PsfUtil

    using FITSIO, ShiftedArrays, Statistics, Photometry
    #include("PsfJ.jl")
    include("KernelUtil.jl")

    export JwstExampleStars, Reg2Oversam, Oversam2Reg, MedianWithMask

    psfj_path = pwd()

    function log10_scale(arr)

        log10arr = zeros(size(arr))
        log10arr[ arr.>0 ] .= log10.(arr[ arr.>0 ])
        return log10arr
    end

    function Reg2Oversam(img; oversampling=1)
        #=
        makes each pixel n x n pixels (with n=oversampling), makes it such that center remains in center pixel
        No sharpening below the original pixel scale is performed. This function should behave as the inverse of
        oversam2regular(). This function is flux conserving.

        Parameters
        ----------
        img : 2d array, square size with odd length n
            Data or model in regular pixel units of the data.
        oversampling : integer >= 1
            oversampling factor per axis of the output relative to the input img

        Returns
        -------
        img_oversampling : 2d array, square size with odd length ns, with ns = oversampling * n for n odd,
            and ns = oversampling * n - 1 for n even
        =#

        if oversampling == 1
            return img
        end

        img_oversampling = repeat(img, inner=[oversampling, oversampling])

        if oversampling % 2 == 1
            # this is already centered with odd total number of pixels
            return img_oversampling
        else
            # for even number super-sampling half a super-sampled pixel offset needs to be performed
            # we do the shift and cut in random directions such that it averages out
    
            img_soverampling1 = ShiftedArray(img_oversampling, (1,1))
            # and the first column and row need to be removed
            img_soverampling1 = img_soverampling1[2:end,2:end]

    
            img_soverampling2 = ShiftedArray(img_oversampling, (1,-1))
            # and the first row and last column need to be removed
            img_soverampling2 = img_soverampling2[2:end,1:end-1]

            img_soverampling3 = ShiftedArray(img_oversampling, (-1,1))
            # and the first column and last row need to be removed
            img_soverampling3 = img_soverampling3[1:end-1,2:end]
            
            img_soverampling4 = ShiftedArray(img_oversampling, (-1,-1))
            # and the last column and row need to be removed
            img_soverampling4 = img_soverampling4[1:end-1,1:end-1]


            img_oversampling = @. (img_soverampling1 + img_soverampling2 + img_soverampling3 + img_soverampling4) / 2

            return img_oversampling ./ oversampling .^ 2
        end

    end

    function Oversam2Reg(img_oversampling; oversampling::Int=1)

        #=
        Averages the pixel flux such that s x s oversampled pixels result in one pixel, with s = oversampling.
        The routine is designed to keep the centroid in the very centered pixel.

        This function should behave as the inverse of regular2oversampled().

        Parameters
        ----------
        img_oversampling : 2d array, square size with odd length NxN
            Oversampled model
        oversampling : integer >= 1
            oversampling factor per axis of the input relative to the output

        Returns
        -------
        image_degraded : 2d array, square size with odd length n
            with odd oversampling, n = N / oversampling, else n = (N + 1) / oversampling
            todo: documentation here not accurate
        =#

        ImgDegraded = KernelUtil.DegradeKernel(img_oversampling, oversampling)

        n = size(img_oversampling)[1]
        # todo: this should be in a single function and not compensating for kernel_util.degrade_kernel()
        if n % oversampling == 0
            n_pix = Int.(n ./ oversampling)
            ImgDegraded = KernelUtil.CutEdges(ImgDegraded, n_pix)
        end

        return ImgDegraded
    end
    

    """
    ...
    imports example stars cutout from early JWST data

    Returns
    -------
    star_list : list of 2d arrays
        cutout stars from JWST data
    ...
    """
    function JwstExampleStars()
        package_path =  psfj_path
        path_stars = string(package_path, "/Data/JWST/")
        star_name = "psf_f090w_star"

        star_list = []
        for i in 0:4
            path = string(path_stars, star_name, string(i), ".fits")

            hdulist_star = FITS(path)

            #star = hdulist_star[1]
            star = read(hdulist_star[1])
            push!(star_list, star)
        end

        return star_list
    
    end

    """
    ...
    pixel-wise median across the different data sets ignoring the masked pixels

    Parameters
    ----------
    data_list : l element of (nx x ny) array of metrix
        set of l equally sized 2d data sets
    mask_list : l element of (nx x ny) array of zeros or ones
        masks applied with 1 = included in the median and 0 = excluded

    Returns
    -------
    median_mask : (nx x ny) array
        pixel-wise median across the 'l' different data sets ignoring the masked pixels
    ...
    """
    function MedianWithMask(data_list, mask_list)
        #mask_list = Array(mask_list)
        x_dim, y_dim = size(data_list[1])
        median_mask = zeros((x_dim, y_dim))
        for i in 1:x_dim
            for j in 1:y_dim
                # slice through same pixels in all kernels
                pixels = [ data_list[k][ i, j] for k in eachindex(data_list)]
                # slice through same pixels in all masks
                masks = [ mask_list[k][ i, j]  for k in eachindex(data_list)]
                # select only those entries that are not masked
                pixel_select = pixels[findall(masks .> 0)]
                # compute median of those values only
                median_mask[i, j] = median(pixel_select)
            end
        end
        return median_mask

    end 

    function FindStars()

        

    end

end
