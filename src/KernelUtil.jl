module KernelUtil

    push!(LOAD_PATH, @__DIR__)

    using ImageTransformations

    export DegradeKernel, CutEdges, Image2Array, Array2Image


    function KernelNorm(kernel::AbstractArray)
        #=
            :param kernel: List of Matrix or Matrix
            :return: normalisation of the psf kernel
        =#
        norm = sum(kernel)
        kernel = kernel ./ norm
        return kernel
    end


    function RotateImage(img::AbstractArray, angle::Real; reshape::Bool=false, fillvalue::Real=0.)
        #=
            Querries ImageTransformations.imrotate routine 
            param img: image to be rotated 
            param angle: angle to be rotated (radian) 
            param reshape: if false, Preserve the original axes

            :return: rotated image.
        =#

        if reshape
            imgR = imrotate(img, angle)
        else
            imgR = imrotate(img, angle, axes(img), fillvalue=fillvalue)
        end

        return imgR
    end

    function Averaging(grid::AbstractArray; numGrid::Int, numPix::Int)
        #=
            Resize 2d pixel grid with numGrid to numPix and averages over the pixels.

            :param grid: higher resolution pixel grid
            :param numGrid: number of pixels per axis in the high resolution input image
            :param numPix: lower number of pixels per axis in the output image (numGrid/numPix
                is integer number)
            :return: averaged pixel grid
        =#

        Nbig = Int(numGrid)
        Nsmall = Int(numPix)
        reshaped = reshape(grid, Nsmall, Int(Nbig / Nsmall), Nsmall, Int(Nbig / Nsmall))
        reshaped = mean(reshaped, dims=4) # size of (Int(Nsmall), Int(Nbig / Nsmall), Int(Nsmall), 1)
        reshaped = mean(reshaped, dims=2) # size of (Int(Nsmall), 1, Int(Nsmall), 1)
        reshaped = reshape(reshaped, Nsmall, Nsmall)# size of (Int(Nsmall), Int(Nsmall))

        return reshaped
    end


    function DegradeKernel(kernel_super::AbstractArray, degrading_factor::Int)
        #=
            :param kernel_super: higher resolution kernel (odd number per axis)
            :param degrading_factor: degrading factor (effectively the super-sampling resolution of the kernel given
            :return: degraded kernel with odd axis number with the sum of the flux/values in the kernel being preserved
        =#

        if degrading_factor == 1
            return kernel_super
        end

        if degrading_factor % 2 == 0
            kernel_low_res = AveragingEvenKernel(kernel_super, degrading_factor)
        else
            kernel_low_res = AveragingOddKernel(kernel_super, degrading_factor)
            # degrading_factor**2  # multiplicative factor added when providing flux conservation
            kernel_low_res = kernel_low_res .* degrading_factor.^2
        end

        return kernel_low_res

    end


    function AveragingEvenKernel(kernel_high_res::AbstractArray, subgrid_res::Int)
        #=
            Makes a lower resolution kernel based 
    
            the kernel_high_res (odd numbers) and
            the subgrid_res (even number), both meant to be centered.

            :param kernel_high_res: high resolution kernel with even subsampling resolution,
                centered
            :param subgrid_res: subsampling resolution (even number)
            :return: averaged undersampling kernel
        =#

        n_kernel_high_res = size(kernel_high_res)[1]
        n_low = @. Int(round(n_kernel_high_res / subgrid_res + 0.5))

        if n_low % 2 == 0
            n_low = n_low + 1
        end

        n_high = @. Int(n_low * subgrid_res - 1)

        @assert n_high % 2 == 1

        if n_high == n_kernel_high_res
            kernel_high_res_edges = kernel_high_res
        else
            i_start = @. Int((n_high - n_kernel_high_res) / 2)
            kernel_high_res_edges = zeros((n_high, n_high))
            kernel_high_res_edges[i_start:end-i_start+1, i_start:end-i_start+1] .= kernel_high_res
        end

        kernel_low_res = zeros((n_low, n_low))
        # adding pixels that are fully within a single re-binned pixel
        for i in 1:subgrid_res-1
            for j in 1:subgrid_res-1
                kernel_low_res = kernel_low_res .+ kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end] 
            end
        end

        # adding half of a pixel that has over-lap with two pixels
        i = subgrid_res
        for j in 1:subgrid_res-1
            kernel_low_res[2:end, :] .= kernel_low_res[2:end, :] .+ kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end] ./ 2
            
            kernel_low_res[1:end-1, :] .+= kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end] ./ 2

        end

        j = subgrid_res
        for i in 1:subgrid_res-1
            kernel_low_res[:, 2:end] .+= (
                kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end] / 2
            )
            kernel_low_res[:, 1:end-1] .+= (
                kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end] / 2
            )
        end

        # adding a quarter of a pixel value that is at the boarder of four pixels
        i = subgrid_res 
        j = subgrid_res 
        kernel_edge = kernel_high_res_edges[i:subgrid_res:end, j:subgrid_res:end]
        kernel_low_res[2:end, 2:end] = kernel_low_res[2:end, 2:end] .+ kernel_edge ./ 4
        kernel_low_res[1:end-1, 2:end] = kernel_low_res[1:end-1, 2:end] .+ kernel_edge ./ 4
        kernel_low_res[2:end, 1:end-1] = kernel_low_res[2:end, 1:end-1] .+ kernel_edge ./ 4
        kernel_low_res[1:end-1, 1:end-1] = kernel_low_res[1:end-1, 1:end-1] .+ kernel_edge ./ 4

        return kernel_low_res ./ subgrid_res.^2
    
    end

    function AveragingOddKernel(kernel_super::AbstractArray, degrading_factor::Real)
        #=
            Makes a lower resolution kernel based 
    
            the kernel_high_res (odd numbers) and
            the subgrid_res (even number), both meant to be centered.

            :param kernel_high_res: high resolution kernel with even subsampling resolution,
                centered
            :param subgrid_res: subsampling resolution (even number)
            :return: averaged undersampling kernel
        =#
        
        n_kernel = size(kernel_super)[1]
        numPix = Int.(round.(n_kernel ./ degrading_factor .+ 0.5))
        if numPix % 2 == 0
            numPix += 1
        end

        n_high = Int.(numPix .* degrading_factor)

        kernel_super_ = zeros((n_high, n_high))
        i_start = Int.((n_high .- n_kernel) ./ 2 )
        kernel_super_[i_start : i_start + n_kernel - 1, i_start : i_start + n_kernel - 1] .= kernel_super
        kernel_low_res = Averaging(kernel_super_, numGrid=n_high, numPix=numPix)

        return kernel_low_res
    
    end


    function CutEdges(image::AbstractArray, num_pix::Int)
        #=
            Cuts out the edges of a 2d image and returns re-sized image to numPix center is
            well defined for odd pixel sizes.

            :param image: 2d numpy array
            :param num_pix: square size of cut out image
            :return: cutout image with size numPix
        =#

        nx, ny = size(image)
        if nx < num_pix || ny < num_pix
            error(
                string("image can not be resized, in routine cut_edges with image shape ("
                ,nx," " ,ny,")and desired new shape (",num_pix," ", num_pix,")")
                )
        end

        if (nx % 2 == 0 && ny % 2 == 1) || (nx % 2 == 1 && ny % 2 == 0)
            error(string("image with odd and even axis (", nx," ", ny,") not supported for re-sizing"))
        end

        if (nx % 2 == 0 && num_pix % 2 == 1) || (nx % 2 == 1 && num_pix % 2 == 0)
            error("image can only be re-sized from even to even or odd to odd number.")
        end

        x_min = Int((nx - num_pix) / 2 + 1) 
        y_min = Int((ny - num_pix) / 2 + 1)
        x_max = Int(nx - x_min + 1)
        y_max = Int(ny - y_min + 1)
        resized = image[x_min:x_max, y_min:y_max]
        return deepcopy(resized)
    end


    function MakeGrid(numPix::Union{Real, AbstractArray}; deltapix::Real=1, subgrid_res::Real=1, left_lower::Bool=false)
        #=
            Creates pixel grid (in 1d arrays of x- and y- positions) default coordinate frame
            is such that (0,0) is in the center of the coordinate grid.

            :param numPix: number of pixels per axis Give an integers for a square grid, or a
                2-length sequence (first, second axis length) for a non-square grid.
            :param deltapix: pixel size
            :param subgrid_res: sub-pixel resolution (default=1)
            :return: x, y position information in two 1d arrays
        =#

        # Check numPix is an integer, or 2-sequence of integers
        if isa(numPix , Union{Tuple, Array})
            
            @assert length(numPix) == 2

            if any(x != round(x) for x in numPix)
                error(string("numPix contains non-integers: ", numPix))
            end
            numPix = Int.(numPix)
        else
            if numPix != round(numPix)
                error(string("Attempt to specify non-int numPix: ", numPix))
            end
            numPix = Int.([numPix, numPix])
        end

        # Super-resolution sampling
        numPix_eff = Int.(numPix .* subgrid_res)
        deltapix_eff = deltapix ./ float.(subgrid_res)

        # Compute unshifted grids.
        # X values change quickly, Y values are repeated many times
        x_grid = repeat(1:numPix_eff[1], outer=numPix_eff[2]) .* deltapix_eff #repeat outer as np.tile
        y_grid = repeat(1:numPix_eff[2], inner=numPix_eff[1]) .* deltapix_eff # repeat inner as np.repeat

        if left_lower == true
            # Shift so (0, 0) is in the "lower left"
            # Note this does not shift when subgrid_res = 1
            shift = @. -1.0 / 2 + 1.0 / (2 * subgrid_res) * [1, 1]
        else
            # Shift so (0, 0) is centered
            shift = @. deltapix_eff * (numPix_eff - 1) / 2
        end

        return x_grid .- shift[1] .- 1., y_grid .- shift[2] .- 1.

    end


    function CutPsf(psf_data::AbstractArray, psf_size::Real, normalisation::Bool=true)
        #=
            Cut the psf properly.

            :param psf_data: image of PSF
            :param psf_size: size of psf
            :return: re-sized and re-normalized PSF
        =#
        kernel = CutEdges(psf_data, psf_size)
        if normalisation == true
            kernel = KernelNorm(kernel)
        end
        return kernel
    end


    function Array2Image(array::AbstractArray, nx::Real=0, ny::Real=0)
        #=
            Returns the information contained in a 1d array into an n*n 2d array (only works
            when length of array is n**2, or nx and ny are provided)

            :param array: image values
            :type array: array of size n**2
            :returns: 2d array
            :raises: AttributeError, KeyError
        =#
        if nx == 0 || ny == 0
            n = Int.(sqrt.(length(array)))
            if n^2 != length(array)
                error(
                    string("lenght of input array given as ", 
                    length(array), " is not square of integer number!")
                    )
            end
            nx, ny = n, n
        end

        image = reshape(array, Int(nx), Int(ny))
        return image

    end


    function Image2Array(image::AbstractArray)
        #=
            Returns the information contained in a 2d array into an n*n 1d array.

            :param image: image values
            :type image: array of size (n,n)
            :returns: 1d array
            :raises: AttributeError, KeyError
        =#

        nx, ny = size(image)  # find the size of the array
        imgh = reshape(image, nx * ny)  # change the shape to be 1d
        return imgh

    end

end;