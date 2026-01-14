import logging

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

log = logging.getLogger(__name__)


def get_deprojected_radii(ra, dec, incl, posang, wcs, shape):
    """
    Calculate deprojected radii and projected angles for a galaxy disk.
    
    Parameters
    ----------
    ra : float
        Center RA in degrees
    dec : float
        Center Dec in degrees
    incl : float
        Inclination angle in degrees (0=face-on)
    posang : float
        Position angle in degrees (North->East)
    wcs : astropy.wcs.WCS
        WCS of the image
    shape : tuple
        Shape of the image (naxis2, naxis1)
        
    Returns
    -------
    radius_deg : ndarray
        Deprojected radius in degrees
    projang_deg : ndarray
        Projected angle in degrees
    """
    # Create pixel coordinate grids
    naxis2, naxis1 = shape
    y, x = np.mgrid[:naxis2, :naxis1]
    
    # Convert to world coordinates
    coords = wcs.celestial.wcs_pix2world(x, y, 0)
    ra_deg = coords[0]
    dec_deg = coords[1]
    
    # Calculate offsets from center
    dx_deg = (ra_deg - ra) * np.cos(np.deg2rad(dec))
    dy_deg = dec_deg - dec
    
    # Rotation angle (rotate x-axis up to major axis)
    rotangle = np.pi/2 - np.deg2rad(posang)
    
    # Create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) + dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) - dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl))
    
    # Calculate deprojected distance from center
    radius_deg = np.sqrt(deprojdx_deg**2 + deprojdy_deg**2)
    
    # Calculate angle with respect to position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))
    
    return radius_deg, projang_deg


    
def get_galaxy_specs(fname_table, galaxy='ngc0628'):
    """
    Get galaxy properties from the merged galaxy table.
    
    Parameters
    ----------
    galaxy : str
        Galaxy name (case-insensitive)
    fname_table : str
        Path to the merged galaxy table
    Returns
    -------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    posang : float
        Position angle in degrees
    incl : float
        Inclination angle in degrees
    reff : float
        Effective radius in arcsec
    """
    import pandas as pd
    
    # Read the merged table (raise error if something went wrong)
    try:
        df = pd.read_csv(fname_table, comment='#')
    except Exception as e:
        log.error(f'Error reading galaxy table {fname_table}: {e}')
        raise e
    
    # Convert galaxy name to lowercase for case-insensitive matching
    galaxy = galaxy.lower()

    # Check that the table has the required columns, and that the type of data in 
    # each column is correct (float for all except name)
    required_columns = ['name', 'ra', 'dec', 'posang', 'inclination', 'size_reff']
    for column in required_columns:
        if column not in df.columns:
            log.error(f'Column {column} not found in table {fname_table}')
            raise ValueError(f'Column {column} not found in table {fname_table}')
        if column == 'name' and not isinstance(df[column].iloc[0], str):
            log.error(f'Column {column} has incorrect type in table {fname_table} (expected string)')
            raise ValueError(f'Column {column} has incorrect type in table {fname_table} (expected string)')
        if column != 'name' and not isinstance(df[column].iloc[0], (float)):
            log.error(f'Column {column} has incorrect type in table {fname_table} (expected float)')
            raise ValueError(f'Column {column} has incorrect type in table {fname_table} (expected float)')

    # Make sure galaxy names in table are lowercase for case-insensitive matching
    df['name'] = df['name'].str.lower()

    row = df[df['name'] == galaxy].iloc[0]
    if len(row) == 0:
        log.error(f'Galaxy {galaxy} not found in table {fname_table}')
        raise ValueError(f'Galaxy {galaxy} not found in table {fname_table}')

    ra = row['ra']
    dec = row['dec']
    posang = row['posang']
    incl = row['inclination']
    reff = row['size_reff']

    return ra, dec, posang, incl, reff



def get_intensity_range(fname_image, fname_table, galaxy = 'ngc0628', reff_factor = 2, percentile_range = [15, 60]):
    """
    Get the intensity range within reff_factor * effective radius of the galaxy center.
    
    Parameters
    ----------
    fname_image : str
        Path to the FITS image
    fname_table : str
        Path to the galaxy properties table
    galaxy : str
        Galaxy name (case-insensitive)
    reff_factor : float
        Factor to multiply effective radius by
    percentile_range : list
        [min, max] percentiles to calculate intensity range
        
    Returns
    -------
    tuple
        (xmin, xmax) where:
        - xmin: minimum intensity value (at percentile_range[0])
        - xmax: maximum intensity value (at percentile_range[1])
        These values are used for binning in the anchoring step.
    """
    try:
        # Input image is a JWST image (SCI extension)
        # Load the image
        with fits.open(fname_image) as hdul:
            # Get the SCI extension
            try:
                sci = hdul['SCI']
            except KeyError:
                sci = hdul[0]
            image = sci.data
            header = sci.header
            w = WCS(header)
            shape = image.shape

        ra_deg, dec_deg, posang_deg, incl_deg, reff_arcsec = get_galaxy_specs(fname_table, galaxy)
        radius_deg, projang_deg = get_deprojected_radii(ra_deg, dec_deg, incl_deg, posang_deg, w, shape)
        radius_arcsec = radius_deg * 60 * 60

        # Get the intensity range within reff_factor * r_e of the center
        mask = (radius_arcsec <= reff_factor * reff_arcsec)
        mask = mask & (image != 0) & np.isfinite(image)
        
        # Ensure we have enough valid pixels
        n_valid = np.sum(mask)


        # Get intensity range
        intensity_range = np.nanpercentile(image[mask], percentile_range)
        xmin, xmax = intensity_range

        # Log results
        log.info(f'Intensity range report for {galaxy}:')
        log.info(f'  Intensity range: [{xmin:.3f}, {xmax:.3f}] from {n_valid} pixels')
        log.info(f'  Percentile range: {percentile_range}')
        log.info(f'  Reff factor: {reff_factor} × {reff_arcsec:.1f}" = {reff_factor * reff_arcsec:.1f}"')
        log.info(f'  Image shape: {image.shape}')
        
        # Final validation
        if xmin >= xmax:
            log.error(f'Invalid intensity range: {xmin} >= {xmax}. Using defaults.')
            return 0.25, 2.0
        
        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            log.error(f'Non-finite intensity range: [{xmin}, {xmax}]. Using defaults.')
            return 0.25, 2.0
            
        return xmin, xmax
    
    # Could put error catching specifically at get_galaxy_specs, but this will catch any other errors
    # too so might as well catch them here
    except Exception as e:
        log.error(f'Error calculating intensity range for {galaxy}: {e}. Using defaults.')
        return 0.25, 2.0
