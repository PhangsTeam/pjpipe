from astroquery.gaia import Gaia
from astropy.coordinates import name_resolve
import astropy.units as u
import sys
import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def gaia_query(number=10000, radius=15 * u.arcmin):
    """
    Query Gaia DR3 for a list of targets in the config file

    Keyword arguments:
    number -- number of sources to return (default 10000)
    radius -- radius of search (default 15 arcmin)      
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if len(sys.argv) == 1:
        config_file = script_dir + '/config/config.toml'
        local_file = script_dir + '/config/local.toml'

    elif len(sys.argv) == 2:
        local_file = script_dir + '/config/local.toml'
        config_file = sys.argv[1]

    with open(config_file, 'rb') as f:
        config = tomllib.load(f)

    with open(local_file, 'rb') as f:
        local = tomllib.load(f)


    prop_ids = config['projects']
    for prop_id in prop_ids:
        targets = config['projects'][prop_id]['targets']
        for target in targets:
            print(target)
            try:
                print('Trying to resolve source {0}'.format(target))
                coords = name_resolve.get_icrs_coordinates(target)
            except:
                print('Unable to resolve {0}'.format(target))
                break
            jobstr = "SELECT TOP {0} * FROM gaiadr3.gaia_source\n".format(number)
            jobstr += "WHERE 1=CONTAINS(POINT('ICRS', gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),"
            jobstr += "CIRCLE('ICRS',{0},{1},{2}))\n".format(coords.ra.deg,
                                                            coords.dec.deg,
                                                            radius.to(u.deg).value)
            jobstr += "ORDER by gaiadr3.gaia_source.phot_bp_mean_mag ASC" 
            print("Launching job query to Gaia archive")
            job = Gaia.launch_job_async(jobstr, dump_to_file=False)
            results = job.get_results()
            removelist = []

            # Strip object columns from FITS table
            for col in results.columns:
                if results[col].dtype == 'object':
                    removelist += [col]
            results.remove_columns(removelist)
            results.write('alignment/' + 'Gaia_DR3_{0}.fits'.format(target.upper()), overwrite=True)

if __name__ == '__main__':
    gaia_query()