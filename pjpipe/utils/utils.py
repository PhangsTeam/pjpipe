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
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from photutils.segmentation import detect_threshold, detect_sources
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject.mosaicking.subset_array import ReprojectedArraySubset
from scipy.interpolate import RegularGridInterpolator
from stdatamodels import util
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels.dqflags import pixel

from .. import __version__

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ALLOWED_REPROJECT_FUNCS = [
    "interp",
    "adaptive",
    "exact",
]

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
    "F560W": 1.882,
    "F770W": 2.445,
    "F1000W": 2.982,
    "F1130W": 3.409,
    "F1280W": 3.818,
    "F1500W": 4.436,
    "F1800W": 5.373,
    "F2100W": 6.127,
    "F2550W": 7.300,
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

    # Finally, if we have a string with a 'pix' in there, we need to convert to arcsec. Unless it's not a number!
    # Then just return the value
    if isinstance(value, str):
        if "pix" in value:
            try:
                value = float(value.strip("pix")) * pixel_scale
            except ValueError:
                pass

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
        check_type (str): How to check if background observation. Options are
            - 'parallel_off', which will use the filename to see if it's a parallel observation with NIRCAM
            - 'check_in_name', which will use the observation name to check, matching against 'background_name'.
            - 'filename', which will use the filename
            Defaults to 'parallel_off'
        background_name (str): Name to indicate background observation. Defaults to 'off'.
    """

    # Figure out if we're a background observation or not
    f_type = "sci"
    if check_bgr:

        # If it's a parallel observation (PHANGS-style)
        if check_type == "parallel_off":
            file_split = os.path.split(file)[-1]
            if file_split.split("_")[1][2] == "2":
                f_type = "bgr"

        # If the backgrounds are labelled differently in the target name
        elif check_type == "check_in_name":
            with datamodels.open(file) as im:
                if background_name in im.meta.target.proposer_name.lower():
                    f_type = "bgr"

        # If we want to use some specific files within the science as observations
        elif check_type == "filename":

            if isinstance(background_name, str):
                background_name = [background_name]

            with datamodels.open(file) as im:

                for bg_name in background_name:
                    if bg_name in im.meta.filename.lower():
                        f_type = "bgr"

        else:
            raise Warning(f"check_type {check_type} not known")

    # Pull out data we need from header
    with datamodels.open(file) as im:
        obs_n = im.meta.observation.observation_number
        obs_filter = im.meta.instrument.filter
        obs_date = im.meta.observation.date_beg
        obs_duration = im.meta.exposure.duration

        # Sometimes the observation label is not defined, so have a fallback here
        obs_label = im.meta.observation.observation_label
        if obs_label is not None:
            obs_label = obs_label.lower()
        else:
            obs_label = ""

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
        reproject_func="interp",
):
    """Reproject an image to an optimal WCS

    Args:
        file: File to reproject
        optimal_wcs: Optimal WCS for input image stack
        optimal_shape: Optimal shape for input image stack
        hdu_type: Type of HDU. Can either be 'data', 'err', or 'var_rnoise'
        do_sigma_clip: Whether to perform sigma-clipping or not.
            Defaults to False
        stacked_image: Stacked image or not? Defaults to False
        do_level_data: Whether to level between amplifiers or not.
            Defaults to False
        reproject_func: Which reproject function to use. Defaults to 'interp',
            but can also be 'exact' or 'adaptive'
    """

    if reproject_func == "interp":
        r_func = reproject_interp
    elif reproject_func == "exact":
        r_func = reproject_exact
    elif reproject_func == "adaptive":
        r_func = reproject_adaptive
    else:
        raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

    hdu_mapping = {
        "data": "SCI",
        "err": "ERR",
        "var_rnoise": "VAR_RNOISE",
    }

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
            elif hdu_type == "err":
                data = copy.deepcopy(hdu.err)
            elif hdu_type == "var_rnoise":
                data = copy.deepcopy(hdu.var_rnoise)
            else:
                raise Warning(f"Unsure how to deal with hdu_type {hdu_type}")

    else:

        hdu_name = hdu_mapping[hdu_type]

        with fits.open(file) as hdu:
            sci = copy.deepcopy(hdu["SCI"].data)
            data = copy.deepcopy(hdu[hdu_name].data)
            wcs = hdu["SCI"].header
            w_in = WCS(wcs)
        dq_bit_mask = None

    sig_mask = None
    if do_sigma_clip:
        sig_mask = make_source_mask(
            sci,
            mask=dq_bit_mask,
            dilate_size=7,
        )
        sig_mask = sig_mask.astype(int)

    data[data == 0] = np.nan

    # This comes from the astropy reproject routines
    edges = sample_array_edges(data.shape, n_samples=11)[::-1]
    edges_out = optimal_wcs.world_to_pixel(w_in.pixel_to_world(*edges))[::-1]

    # Determine the cutout parameters

    # In some cases, images might not have valid coordinates in the corners,
    # such as all-sky images or full solar disk views. In this case we skip
    # this step and just use the full output WCS for reprojection.

    ndim_out = len(optimal_shape)

    skip_data = False
    if np.any(np.isnan(edges_out)):
        bounds = list(zip([0] * ndim_out, optimal_shape))
    else:
        bounds = []
        for idim in range(ndim_out):
            imin = max(0, int(np.floor(edges_out[idim].min() + 0.5)))
            imax = min(optimal_shape[idim], int(np.ceil(edges_out[idim].max() + 0.5)))
            bounds.append((imin, imax))
            if imax < imin:
                skip_data = True
                break

    if skip_data:
        return

    slice_out = tuple([slice(imin, imax) for (imin, imax) in bounds])

    if isinstance(optimal_wcs, WCS):
        wcs_out_indiv = optimal_wcs[slice_out]
    else:
        wcs_out_indiv = SlicedLowLevelWCS(optimal_wcs.low_level_wcs, slice_out)

    shape_out_indiv = [imax - imin for (imin, imax) in bounds]

    data_reproj_small = r_func(
        (data, wcs),
        output_projection=wcs_out_indiv,
        shape_out=shape_out_indiv,
        return_footprint=False,
    )

    # Mask out bad DQ, but only for unstacked images. This needs to use
    # reproject_interp, so we can keep whole numbers
    if not stacked_image:
        dq_reproj_small = reproject_interp(
            (dq_bit_mask, wcs),
            output_projection=wcs_out_indiv,
            shape_out=shape_out_indiv,
            return_footprint=False,
            order="nearest-neighbor",
        )
        data_reproj_small[dq_reproj_small == 1] = np.nan

    # If we're sigma-clipping, reproject the mask. This needs to use
    # reproject_interp, so we can keep whole numbers
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
        data_reproj_small,
        footprint,
        bounds,
    )

    del hdu
    gc.collect()

    return data_array


def sample_array_edges(shape, *, n_samples):
    # Given an N-dimensional array shape, sample each edge of the array using
    # the requested number of samples (which will include vertices). To do this
    # we iterate through the dimensions and for each one we sample the points
    # in that dimension and iterate over the combination of other vertices.
    # Returns an array with dimensions (N, n_samples)
    all_positions = []
    ndim = len(shape)
    shape = np.array(shape)
    for idim in range(ndim):
        for vertex in range(2 ** ndim):
            positions = -0.5 + shape * ((vertex & (2 ** np.arange(ndim))) > 0).astype(int)
            positions = np.broadcast_to(positions, (n_samples, ndim)).copy()
            positions[:, idim] = np.linspace(-0.5, shape[idim] - 0.5, n_samples)
            all_positions.append(positions)
    positions = np.unique(np.vstack(all_positions), axis=0).T
    return positions


def do_jwst_convolution(
        file_in,
        file_out,
        file_kernel,
        blank_zeros=True,
        output_grid=None,
        reproject_func="interp",
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
        reproject_func: Which reproject function to use. Defaults to 'interp',
            but can also be 'exact' or 'adaptive'
    """

    if reproject_func == "interp":
        r_func = reproject_interp
    elif reproject_func == "exact":
        r_func = reproject_exact
    elif reproject_func == "adaptive":
        r_func = reproject_adaptive
    else:
        raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

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

            repr_data, fp = r_func(
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
            repr_err = r_func(
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


def save_file(im,
              out_name,
              dr_version,
              ):
    """Save out an image, adding in useful metadata

    Args:
        im: Input JWST datamodel
        out_name: File to save output to
        dr_version: Data processing version
    """

    # Save versions both in the metadata, and in fits history
    im.meta.pjpipe_version = __version__
    im.meta.pjpipe_dr_version = dr_version
    entry = util.create_history_entry(f"PJPIPE VER: {__version__}")
    im.history.append(entry)
    entry = util.create_history_entry(f"DATA PROCESSING VER: {dr_version}")
    im.history.append(entry)

    im.save(out_name)

    return True


def make_stacked_image(
        files,
        out_name,
        additional_hdus=None,
        auto_rotate=True,
        reproject_func="interp",
        match_background=False,
):
    """Create a quick stacked image from a series of input images

    Args:
        files: List of input files
        out_name: Output stacked file
        additional_hdus: Can also append some additional data beyond the science
            extension by specifying the fits extension here. Defaults to None,
            which will not add anything extra
        auto_rotate: Whether to rotate the WCS to make a minimum sized image.
            Defaults to True
        reproject_func: Which reproject function to use. Defaults to 'interp',
            but can also be 'exact' or 'adaptive'
        match_background: Whether to match backgrounds when making the stack.
            Defaults to False
    """

    # HDUs we need to square before they go into the reprojection
    sq_hdus = [
        "ERR",
    ]

    # HDUs we'll need to take the square root of the stacked image to be meaningful
    sqrt_hdus = [
        "ERR",
        "VAR_RNOISE",
    ]

    combine_functions = {
        "SCI": "mean",
        "ERR": "sum",
        "VAR_RNOISE": "sum",
    }

    if reproject_func == "interp":
        r_func = reproject_interp
    elif reproject_func == "exact":
        r_func = reproject_exact
    elif reproject_func == "adaptive":
        r_func = reproject_adaptive
    else:
        raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

    if additional_hdus is None:
        additional_hdus = []
    if isinstance(additional_hdus, str):
        additional_hdus = [additional_hdus]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hdus = []

        for file in files:
            hdu = fits.open(file)

            dq_bit_mask = get_dq_bit_mask(hdu["DQ"].data)

            # For some reason this doesn't like being in-place edited,
            # so dipsy-doodle
            data = copy.deepcopy(hdu["SCI"].data)
            data[dq_bit_mask != 0] = np.nan
            hdu["SCI"].data = copy.deepcopy(data)

            for additional_hdu in additional_hdus:

                data = copy.deepcopy(hdu[additional_hdu].data)
                data[dq_bit_mask != 0] = np.nan
                hdu[additional_hdu].data = copy.deepcopy(data)

                # Make sure the full WCS is in there by copying over the header
                hdr = copy.deepcopy(hdu["SCI"].header)
                hdr["EXTNAME"] = additional_hdu
                hdu[additional_hdu].header = copy.deepcopy(hdr)

                if additional_hdu in sq_hdus:
                    hdu[additional_hdu].data = hdu[additional_hdu].data ** 2

            hdus.append(hdu)

        output_projection, shape_out = find_optimal_celestial_wcs(hdus,
                                                                  hdu_in="SCI",
                                                                  auto_rotate=auto_rotate,
                                                                  )
        hdr = output_projection.to_header()

        # Loop over the various HDUs we want to reproject
        stacked_images = {}
        stacked_image, stacked_footprint = reproject_and_coadd(
            hdus,
            output_projection=output_projection,
            shape_out=shape_out,
            hdu_in="SCI",
            combine_function=combine_functions["SCI"],
            reproject_function=r_func,
            match_background=match_background,
        )
        stacked_image[stacked_footprint == 0] = np.nan
        stacked_images["SCI"] = copy.deepcopy(stacked_image)

        for additional_hdu in additional_hdus:
            stacked_image, stacked_footprint = reproject_and_coadd(
                hdus,
                output_projection=output_projection,
                shape_out=shape_out,
                hdu_in=additional_hdu,
                combine_function=combine_functions[additional_hdu],
                reproject_function=r_func,
                match_background=match_background,
            )
            stacked_image[stacked_footprint == 0] = np.nan
            if additional_hdu in sqrt_hdus:
                stacked_image = np.sqrt(stacked_image)

            stacked_images[additional_hdu] = copy.deepcopy(stacked_image)

        # Create an HDU list here
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU(header=hdus[0][0].header))
        for key in stacked_images:
            hdu.append(fits.ImageHDU(data=stacked_images[key], header=hdr, name=key))

        hdu.writeto(
            out_name,
            overwrite=True,
        )

        del hdus
        gc.collect()

    return True
