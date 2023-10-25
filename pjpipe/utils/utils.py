import copy
import functools
import gc
import inspect
import logging
import os
import warnings

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table
from astropy.wcs import WCS
from photutils.segmentation import detect_threshold, detect_sources
from reproject import reproject_interp
from reproject.mosaicking.subset_array import ReprojectedArraySubset
from scipy.interpolate import RegularGridInterpolator
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels.dqflags import pixel

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Useful values

PIXEL_SCALE_NAMES = ["XPIXSIZE", "CDELT1", "CD1_1", "PIXELSCL"]

# Pixel scales
jwst_pixel_scales = {
    "miri": 0.11,
    "nircam_long": 0.063,
    "nircam_short": 0.031,
}

# All NIRCAM bands
nircam_bands = [
    "F070W",
    "F090W",
    "F115W",
    "F140M",
    "F150W",
    "F162M",
    "F164N",
    "F150W2",
    "F182M",
    "F187N",
    "F200W",
    "F210M",
    "F212N",
    "F250M",
    "F277W",
    "F300M",
    "F322W2",
    "F323N",
    "F335M",
    "F356W",
    "F360M",
    "F405N",
    "F410M",
    "F430M",
    "F444W",
    "F460M",
    "F466N",
    "F470N",
    "F480M",
]

# All MIRI bands
miri_bands = [
    "F560W",
    "F770W",
    "F1000W",
    "F1130W",
    "F1280W",
    "F1500W",
    "F1800W",
    "F2100W",
    "F2550W",
]

# FWHM of bands in pixels
fwhms_pix = {
    # NIRCAM
    "F070W": 0.987,
    "F090W": 1.103,
    "F115W": 1.298,
    "F140M": 1.553,
    "F150W": 1.628,
    "F162M": 1.770,
    "F164N": 1.801,
    "F150W2": 1.494,
    "F182M": 1.990,
    "F187N": 2.060,
    "F200W": 2.141,
    "F210M": 2.304,
    "F212N": 2.341,
    "F250M": 1.340,
    "F277W": 1.444,
    "F300M": 1.585,
    "F322W2": 1.547,
    "F323N": 1.711,
    "F335M": 1.760,
    "F356W": 1.830,
    "F360M": 1.901,
    "F405N": 2.165,
    "F410M": 2.179,
    "F430M": 2.300,
    "F444W": 2.302,
    "F460M": 2.459,
    "F466N": 2.507,
    "F470N": 2.535,
    "F480M": 2.574,
    # MIRI
    "F560W": 1.636,
    "F770W": 2.187,
    "F1000W": 2.888,
    "F1130W": 3.318,
    "F1280W": 3.713,
    "F1500W": 4.354,
    "F1800W": 5.224,
    "F2100W": 5.989,
    "F2550W": 7.312,
}

band_exts = {
    "nircam": "nrc*",
    "miri": "mirimage",
}

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def get_pixscale(hdu):
    """Get pixel scale from header.

    Checks HDU header and returns a pixel scale

    Args:
        hdu: hdu to get pixel scale for
    """

    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(hdu.header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ["CDELT1", "CD1_1"]:
                pix_scale = WCS(hdu.header).proj_plane_pixel_scales()[0].value * 3600
                # pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning("No pixel scale found")


def load_toml(filename):
    """Open a .toml file

    Args:
        filename (str): Path to toml file
    """

    with open(filename, "rb") as f:
        toml_dict = tomllib.load(f)

    return toml_dict


def get_band_type(
        band,
        short_long_nircam=False,
):
    """Get the instrument type from the band name

    Args:
        band (str): Name of band
        short_long_nircam (bool): Whether to distinguish between short/long
            NIRCam bands. Defaults to False
    """

    if band in miri_bands:
        band_type = "miri"
    elif band in nircam_bands:
        band_type = "nircam"
    else:
        raise ValueError(f"band {band} unknown")

    if not short_long_nircam:
        return band_type

    else:
        if band_type in ["nircam"]:
            if int(band[1:4]) <= 212:
                short_long = "nircam_short"
            else:
                short_long = "nircam_long"
            band_type = "nircam"
        else:
            short_long = copy.deepcopy(band_type)

        return band_type, short_long


def get_band_ext(band):
    """Get the specific extension (e.g. mirimage) for a band"""

    band_type = get_band_type(band)
    band_ext = band_exts[band_type]

    return band_ext


def get_default_args(func):
    """Pull the default arguments from a function"""

    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_kws(
        parameters,
        func,
        band,
        target,
        max_level=None,
):
    """Set up kwarg dict for a function, looping over band and target

    Args:
        parameters: Dictionary of parameters
        func: Function to set the parameters for
        band: Band to pull band-specific parameters for
        target: Target to pull target-specific parameters for
        max_level: How far to recurse down the dictionary. Defaults
            to None, which will recurse all the way down
    """

    args = get_default_args(func)

    func_kws = {}
    for arg in args:
        if arg in parameters:
            arg_val = parse_parameter_dict(
                parameters=parameters,
                key=arg,
                band=band,
                target=target,
                max_level=max_level,
            )
            if arg_val == "VAL_NOT_FOUND":
                arg_val = args[arg]
        else:
            arg_val = args[arg]

        func_kws[arg] = arg_val

    return func_kws


def parse_parameter_dict(
        parameters,
        key,
        band,
        target,
        max_level=None,
):
    """Pull values out of a parameter dictionary

    Args:
        parameters (dict): Dictionary of parameters and associated values
        key (str): Particular key in parameter_dict to consider
        band (str): JWST band, to parse out band type and potentially per-band
            values
        target (str): JWST target, for very specific values
        max_level: Maximum level to recurse down. Defaults to None, which will
            go until it finds something that's not a dictionary
    """

    if max_level is None:
        max_level = np.inf

    value = parameters[key]

    band_type, short_long = get_band_type(
        band,
        short_long_nircam=True,
    )

    pixel_scale = jwst_pixel_scales[short_long]

    found_value = False
    level = 0

    while level < max_level and not found_value:
        if isinstance(value, dict):
            # Define a priority here. It goes:
            # * target
            # * band
            # * nircam_short/nircam_long
            # * nircam/miri

            if target in value:
                value = value[target]

            elif band in value:
                value = value[band]

            elif band_type == "nircam" and short_long in value:
                value = value[short_long]

            elif band_type in value:
                value = value[band_type]

            else:
                value = "VAL_NOT_FOUND"

            level += 1

        if not isinstance(value, dict):
            found_value = True

    # Finally, if we have a string with a 'pix' in there, we need to convert to arcsec
    if isinstance(value, str):
        if "pix" in value:
            value = float(value.strip("pix")) * pixel_scale

    return value


def attribute_setter(
        pipeobj,
        parameters,
        band,
        target,
):
    """Set attributes for a function

    Args:
        pipeobj: Function/class to set parameters for
        parameters: Dictionary of parameters to set
        band: Band to pull band-specific parameters for
        target: Target to pull target-specific parameters for
    """

    for key in parameters.keys():
        if type(parameters[key]) is dict:
            for subkey in parameters[key]:
                value = parse_parameter_dict(
                    parameters=parameters[key],
                    key=subkey,
                    band=band,
                    target=target,
                )
                if value == "VAL_NOT_FOUND":
                    continue

                recursive_setattr(
                    pipeobj,
                    ".".join([key, subkey]),
                    value,
                )

        else:
            value = parse_parameter_dict(
                parameters=parameters,
                key=key,
                band=band,
                target=target,
            )
            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(
                pipeobj,
                key,
                value,
            )
    return pipeobj


def recursive_setattr(
        f,
        attribute,
        value,
        protected=False,
):
    """Set potentially recursive function attributes.

    This is needed for the JWST pipeline steps, which have levels to them

    Args:
        f: Function to consider
        attribute: Attribute to consider
        value: Value to set
        protected: If a function is protected, this won't strip out the leading underscore
    """

    pre, _, post = attribute.rpartition(".")

    if pre:
        pre_exists = True
    else:
        pre_exists = False

    if protected:
        post = "_" + post
    return setattr(recursive_getattr(f, pre) if pre_exists else f, post, value)


def recursive_getattr(
        f,
        attribute,
        *args,
):
    """Get potentially recursive function attributes.

    This is needed for the JWST pipeline steps, which have levels to them

    Args:
        f: Function to consider
        attribute: Attribute to consider
        args: Named arguments
    """

    def _getattr(f, attribute):
        return getattr(f, attribute, *args)

    return functools.reduce(_getattr, [f] + attribute.split("."))


def get_obs_table(
        files,
        check_bgr=False,
        check_type="parallel_off",
        background_name="off",
):
    """Pull necessary info out of fits headers"""

    tab = Table(
        names=[
            "File",
            "Type",
            "Obs_ID",
            "Filter",
            "Start",
            "Exptime",
            "Objname",
            "Program",
            "Array",
        ],
        dtype=[
            str,
            str,
            str,
            str,
            str,
            float,
            str,
            str,
            str,
        ],
    )

    for f in files:
        tab.add_row(
            parse_fits_to_table(
                f,
                check_bgr=check_bgr,
                check_type=check_type,
                background_name=background_name,
            )
        )

    return tab


def parse_fits_to_table(
        file,
        check_bgr=False,
        check_type="parallel_off",
        background_name="off",
):
    """Pull necessary info out of fits headers

    Args:
        file (str): File to get info for
        check_bgr (bool): Whether to check if this is a science or background observation (in the MIRI case)
        check_type (str): How to check if background observation. Options are 'parallel_off', which will use the
            filename to see if it's a parallel observation with NIRCAM, or 'check_in_name', which will use the
            observation name to check, matching against 'background_name'. Defaults to 'parallel_off'
        background_name (str): Name to indicate background observation. Defaults to 'off'.
    """

    # Figure out if we're a background observation or not
    f_type = "sci"
    if check_bgr:
        if check_type == "parallel_off":
            file_split = os.path.split(file)[-1]
            if file_split.split("_")[1][2] == "2":
                f_type = "bgr"

        elif check_type == "check_in_name":
            with datamodels.open(file) as im:
                if background_name in im.meta.target.proposer_name.lower():
                    f_type = "bgr"

        else:
            raise Warning(f"check_type {check_type} not known")

    # Pull out data we need from header
    with datamodels.open(file) as im:
        obs_n = im.meta.observation.observation_number
        obs_filter = im.meta.instrument.filter
        obs_date = im.meta.observation.date_beg
        obs_duration = im.meta.exposure.duration
        obs_label = im.meta.observation.observation_label.lower()
        obs_program = im.meta.observation.program_number
        array_name = im.meta.subarray.name.lower().strip()

    return (
        file,
        f_type,
        obs_n,
        obs_filter,
        obs_date,
        obs_duration,
        obs_label,
        obs_program,
        array_name,
    )


def get_dq_bit_mask(
    dq,
    bit_flags="~DO_NOT_USE+NON_SCIENCE",
):
    """Get a DQ bit mask from an input image

    Args:
        dq: DQ array
        bit_flags: Bit flags to get mask for. Defaults to only get science pixels
    """

    dq_bits = interpret_bit_flags(bit_flags=bit_flags, flag_name_map=pixel)

    dq_bit_mask = bitfield_to_boolean_mask(
        dq.astype(np.uint8), dq_bits, good_mask_value=0, dtype=np.uint8
    )

    return dq_bit_mask


def make_source_mask(
        data,
        mask=None,
        nsigma=3,
        npixels=3,
        dilate_size=11,
        sigclip_iters=5,
):
    """Make a source mask from segmentation image"""

    sc = SigmaClip(
        sigma=nsigma,
        maxiters=sigclip_iters,
    )
    threshold = detect_threshold(
        data,
        mask=mask,
        nsigma=nsigma,
        sigma_clip=sc,
    )

    segment_map = detect_sources(
        data,
        threshold,
        npixels=npixels,
    )

    # If sources are detected, we can make a segmentation mask, else fall back to 0 array
    try:
        mask = segment_map.make_source_mask(size=dilate_size)
    except AttributeError:
        mask = np.zeros(data.shape, dtype=bool)

    return mask


def sigma_clip(
        data,
        dq_mask=None,
        sigma=1.5,
        n_pixels=5,
        max_iterations=20,
):
    """Get sigma-clipped statistics for data"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = make_source_mask(data, mask=dq_mask, nsigma=sigma, npixels=n_pixels)
        if dq_mask is not None:
            mask = np.logical_or(mask, dq_mask)
        mean, median, std_dev = sigma_clipped_stats(
            data, mask=mask, sigma=sigma, maxiters=max_iterations
        )

    return mean, median, std_dev


def reproject_image(
        file,
        optimal_wcs,
        optimal_shape,
        hdu_type="data",
        do_sigma_clip=False,
        stacked_image=False,
        do_level_data=False,
):
    """Reproject an image to an optimal WCS

    Args:
        file: File to reproject
        optimal_wcs: Optimal WCS for input image stack
        optimal_shape: Optimal shape for input image stack
        hdu_type: Type of HDU. Can either be 'data' or 'var_rnoise'
        do_sigma_clip: Whether to perform sigma-clipping or not.
            Defaults to False
        stacked_image: Stacked image or not? Defaults to False
        do_level_data: Whether to level between amplifiers or not.
            Defaults to False
    """

    if not stacked_image:
        with datamodels.open(file) as hdu:

            dq_bit_mask = get_dq_bit_mask(hdu.dq)

            wcs = hdu.meta.wcs.to_fits_sip()
            w_in = WCS(wcs)

            # Level data (but not in subarray mode)
            if "sub" not in hdu.meta.subarray.name.lower() and do_level_data and hdu_type == "data":
                hdu.data = level_data(hdu)

            if hdu_type == "data":
                data = copy.deepcopy(hdu.data)
            elif hdu_type == "var_rnoise":
                data = copy.deepcopy(hdu.var_rnoise)
            else:
                raise Warning(f"Unsure how to deal with hdu_type {hdu_type}")

    else:
        with fits.open(file) as hdu:
            data = copy.deepcopy(hdu["SCI"].data)
            wcs = hdu["SCI"].header
            w_in = WCS(wcs)
        dq_bit_mask = None

    sig_mask = None
    if do_sigma_clip:
        sig_mask = make_source_mask(
            data,
            mask=dq_bit_mask,
            dilate_size=7,
        )
        sig_mask = sig_mask.astype(int)

    data[data == 0] = np.nan

    # Find the minimal shape for the reprojection. This is from the astropy reproject routines
    ny, nx = data.shape
    xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    xc_out, yc_out = optimal_wcs.world_to_pixel(w_in.pixel_to_world(xc, yc))

    if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
        imin = 0
        imax = optimal_shape[1]
        jmin = 0
        jmax = optimal_shape[0]
    else:
        imin = max(0, int(np.floor(xc_out.min() + 0.5)))
        imax = min(optimal_shape[1], int(np.ceil(xc_out.max() + 0.5)))
        jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
        jmax = min(optimal_shape[0], int(np.ceil(yc_out.max() + 0.5)))

    if imax < imin or jmax < jmin:
        return

    wcs_out_indiv = optimal_wcs[jmin:jmax, imin:imax]
    shape_out_indiv = (jmax - jmin, imax - imin)

    data_reproj_small = reproject_interp(
        (data, wcs),
        output_projection=wcs_out_indiv,
        shape_out=shape_out_indiv,
        return_footprint=False,
    )

    # Mask out bad DQ, but only for unstacked images
    if not stacked_image:
        dq_reproj_small = reproject_interp(
            (dq_bit_mask, wcs),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            return_footprint=False,
            order="nearest-neighbor",
        )
        data_reproj_small[dq_reproj_small == 1] = np.nan

    if do_sigma_clip:
        sig_mask_reproj_small = reproject_interp(
            (sig_mask, wcs),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            return_footprint=False,
            order="nearest-neighbor",
        )
        data_reproj_small[sig_mask_reproj_small == 1] = np.nan

    footprint = np.ones_like(data_reproj_small)
    footprint[
        np.logical_or(data_reproj_small == 0, ~np.isfinite(data_reproj_small))
    ] = 0
    data_array = ReprojectedArraySubset(
        data_reproj_small, footprint, imin, imax, jmin, jmax
    )

    del hdu
    gc.collect()

    return data_array


def do_jwst_convolution(
        file_in,
        file_out,
        file_kernel,
        blank_zeros=True,
        output_grid=None,
):
    """
    Convolves input image with an input kernel, and writes to disk.

    Will also process errors and do reprojection, if specified

    Args:
        file_in: Path to image file
        file_out: Path to output file
        file_kernel: Path to kernel for convolution
        blank_zeros: If True, then all zero values will be set to NaNs. Defaults to True
        output_grid: None (no reprojection to be done) or tuple (wcs, shape) defining the grid for reprojection.
            Defaults to None

    """
    with fits.open(file_kernel) as kernel_hdu:
        kernel_pix_scale = get_pixscale(kernel_hdu[0])
        # Note the shape and grid of the kernel as input
        kernel_data = kernel_hdu[0].data
        kernel_hdu_length = kernel_hdu[0].data.shape[0]
        original_central_pixel = (kernel_hdu_length - 1) / 2
        original_grid = (
                                np.arange(kernel_hdu_length) - original_central_pixel
                        ) * kernel_pix_scale

    with fits.open(file_in) as image_hdu:
        if blank_zeros:
            # make sure that all zero values were set to NaNs, which
            # astropy convolution handles with interpolation
            image_hdu["ERR"].data[(image_hdu["SCI"].data == 0)] = np.nan
            image_hdu["SCI"].data[(image_hdu["SCI"].data == 0)] = np.nan

        image_pix_scale = get_pixscale(image_hdu["SCI"])

        # Calculate kernel size after interpolating to the image pixel
        # scale. Because sometimes there's a little pixel scale rounding
        # error, subtract a little bit off the optimum size (Tom
        # Williams).

        interpolate_kernel_size = (
                np.floor(kernel_hdu_length * kernel_pix_scale / image_pix_scale) - 2
        )

        # Ensure the kernel has a central pixel

        if interpolate_kernel_size % 2 == 0:
            interpolate_kernel_size -= 1

        # Define a new coordinate grid onto which to project the kernel
        # but using the pixel scale of the image

        new_central_pixel = (interpolate_kernel_size - 1) / 2
        new_grid = (
                           np.arange(interpolate_kernel_size) - new_central_pixel
                   ) * image_pix_scale
        x_coords_new, y_coords_new = np.meshgrid(new_grid, new_grid)

        # Do the reprojection from the original kernel grid onto the new
        # grid with pixel scale matched to the image

        grid_interpolated = RegularGridInterpolator(
            (original_grid, original_grid),
            kernel_data,
            bounds_error=False,
            fill_value=0.0,
        )
        kernel_interp = grid_interpolated(
            (x_coords_new.flatten(), y_coords_new.flatten())
        )
        kernel_interp = kernel_interp.reshape(x_coords_new.shape)

        # Ensure the interpolated kernel is normalized to 1
        kernel_interp = kernel_interp / np.nansum(kernel_interp)

        # Now with the kernel centered and matched in pixel scale to the
        # input image use the FFT convolution routine from astropy to
        # convolve.

        conv_im = convolve_fft(
            image_hdu["SCI"].data,
            kernel_interp,
            allow_huge=True,
            preserve_nan=True,
            fill_value=np.nan,
        )

        # Convolve errors (with kernel**2, do not normalize it).
        # This, however, doesn't account for covariance between pixels
        conv_err = np.sqrt(
            convolve_fft(
                image_hdu["ERR"].data ** 2,
                kernel_interp ** 2,
                preserve_nan=True,
                allow_huge=True,
                normalize_kernel=False,
            )
        )

        image_hdu["SCI"].data = conv_im
        image_hdu["ERR"].data = conv_err

        if output_grid is None:
            image_hdu.writeto(file_out, overwrite=True)
        else:
            # Reprojection to target wcs grid define in output_grid
            target_wcs, target_shape = output_grid
            hdulist_out = fits.HDUList([fits.PrimaryHDU(header=image_hdu[0].header)])

            repr_data, fp = reproject_interp(
                (conv_im, image_hdu["SCI"].header),
                output_projection=target_wcs,
                shape_out=target_shape,
            )
            fp = fp.astype(bool)
            repr_data[~fp] = np.nan
            header = image_hdu["SCI"].header
            header.update(target_wcs.to_header())
            hdulist_out.append(fits.ImageHDU(data=repr_data, header=header, name="SCI"))

            # Note - this ignores the errors of interpolation and thus the resulting errors might be underestimated
            repr_err = reproject_interp(
                (conv_err, image_hdu["SCI"].header),
                output_projection=target_wcs,
                shape_out=target_shape,
                return_footprint=False,
            )
            repr_err[~fp] = np.nan
            header = image_hdu["ERR"].header
            hdulist_out.append(fits.ImageHDU(data=repr_err, header=header, name="ERR"))

            hdulist_out.writeto(file_out, overwrite=True)


def level_data(
        im,
):
    """Level overlaps in NIRCAM amplifiers

    Args:
        im: Input datamodel
    """

    data = copy.deepcopy(im.data)

    quadrant_size = data.shape[1] // 4

    dq_mask = get_dq_bit_mask(dq=im.dq)
    dq_mask = dq_mask | ~np.isfinite(im.data) | ~np.isfinite(im.err) | (im.data == 0)

    for i in range(3):
        quad_1 = data[:, i * quadrant_size: (i + 1) * quadrant_size][
                 :, quadrant_size - 20:
                 ]
        dq_1 = dq_mask[:, i * quadrant_size: (i + 1) * quadrant_size][
               :, quadrant_size - 20:
               ]
        quad_2 = data[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size][:, :20]
        dq_2 = dq_mask[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size][:, :20]

        quad_1[dq_1] = np.nan
        quad_2[dq_2] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_1 = np.nanmedian(
                quad_1,
                axis=1,
            )
            med_2 = np.nanmedian(
                quad_2,
                axis=1,
            )
            diff = med_1 - med_2
            delta = sigma_clipped_stats(diff, maxiters=None)[1]

        data[:, (i + 1) * quadrant_size: (i + 2) * quadrant_size] += delta

    return data
