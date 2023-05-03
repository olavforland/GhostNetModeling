# Imports
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import logging; logger = logging.getLogger(__name__)

from pprint import pprint
from datetime import datetime
from typing import Tuple


from opendrift.readers import reader_netCDF_CF_generic, reader_global_landmask
from opendrift.models.plastdrift import PlastDrift


from utils import random_date, load_sim_results, set_seed


class GhostNetSimulator:
    """Class that handles simulation of ghost nets
    """

    def __init__(self, sim_class: PlastDrift=PlastDrift) -> None:
        """Initialize readers and simulation object in Opendrift, and read file of fishing activity.

        Args:
            sim_clas (PlastDrift, optional): Class of object we want to simulate. See Opendrift documentation. Defaults to PlastDrift.
        """
        # Reader for ocean data
        reader_norkyst = reader_netCDF_CF_generic.Reader(
            filename='https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be'
        )

        # Landmask reader
        reader_landmask = reader_global_landmask.Reader()

        self.sim = sim_class()
        # Make sure objects stop when hitting shore
        self.sim.set_config('general:coastline_action', 'stranding')
        # Add readers to sim object
        self.sim.add_reader([reader_landmask, reader_norkyst])

        # Get fishing activity from file
        self.fishing_activity = self._read_fishing_activity(filepath='./data/Fishing_activity_2021-2022.csv')

    def _read_fishing_activity(self, filepath: str) -> pd.DataFrame:
        """Helper for reading fishing activity

        Args:
            filepath (str): Path to file

        Returns:
            pd.DataFrame: Dataframe of fishing activity.
        """

        # Imported data from excel sheets
        fishing_activity = pd.read_csv(filepath, delimiter=';', header=0,usecols=[0,1,3,4], low_memory=False)
        fishing_activity = fishing_activity[fishing_activity['geartype'] == 'set_gillnets']
        fishing_activity = fishing_activity.rename({'Lat': 'lat', 'Lon': 'lon', 'Apparent Fishing hours': 'fishing_hours'}, axis=1)
        fishing_activity['fishing_minutes'] = (fishing_activity['fishing_hours'] * 60).astype(int)
        
        # Only southern Norway (South of Namsos)
        fishing_activity = fishing_activity[fishing_activity['lat'] < 64]

        return fishing_activity
    

    def sample_ghostnets(self, lam:float=1/300) -> Tuple[np.ndarray]:
        """Samples ghost nets based on fishing activity according to Poisson distribution

        Args:
            lam (float, optional): Rate fishing vessels lose fishing gear (per minute). Defaults to 1/300.

        Returns:
            Tuple[np.ndarray]: The sampled latitude and longitudes
        """
        # Samples lost fishing gear from a Poisson distribution based on fishing activity
        fishing_activity = self.fishing_activity.copy(deep=True)
        
        fishing_activity['n_particles'] = np.random.poisson(lam=lam * fishing_activity['fishing_minutes'])

        # fishing_activity = fishing_activity[fishing_activity['fishing_hours'] > 5]
        # fishing_activity['n_particles'] = fishing_activity['fishing_hours']//10+1

        # Repeat n_particles times
        repeated = fishing_activity.loc[fishing_activity.index.repeat(fishing_activity['n_particles'])]
        init_lats = repeated['lat'].values
        init_lons = repeated['lon'].values

        return (init_lats, init_lons)
    

    def run(self, time_step: int=3600, steps: int=24*6, outfile: str='./simulation_results/ghostnet_points.npz') -> None:
        """Run the ghost net simulation at a randdom date by sampling initial points based on fishing activity.

        Args:
            time_step (int, optional): Size of each time step in the simulation. Defaults to 3600 (seconds).
            steps (int, optional): Number of steps to perform. Defaults to 24*6.
            outfile (str, optional): File to save end state. Defaults to './simulation_results/ghostnet_points.npz'.
        """

        # Sample init points
        init_lats, init_lons = self.sample_ghostnets(lam=1/300)
        
        # Since we don't know anything about how the fishing activity is distributed,
        # we simply draw a random data in the interval of the data
        start_date = datetime.strptime('1/1/2021', '%m/%d/%Y')
        end_date = datetime.strptime('1/1/2023', '%m/%d/%Y')
        sim_start = random_date(start=start_date, end=end_date)

        self.sim.seed_elements(lon=init_lons, lat=init_lats, time=sim_start, radius=1000)

        tempfile = './simulation_results/temp.nc'
        self.sim.run(time_step=time_step, steps=steps, outfile=tempfile)

        self._save_simulation_results(infile=tempfile, outfile=outfile)
        os.remove(tempfile)

    def _save_simulation_results(self, infile: str, outfile: str) -> None:
        """Helper for saving simulation results. Firsts checks if there are existing results in outfile. 
        If there are, combines these with the results in infile. Finally writes (combined) results to outfile

        Args:
            infile (str): Temporary file where simulation results are stored
            outfile (str): File to save results to
        """

        # Retrieve data from simulation
        ds = nc.Dataset(infile)
        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        # Retrive initial positions, final activce positions, and final deactivated positions
        init_lons, init_lats = lons[:, 0], lats[:, 0]
        lon_active, lat_active = self.sim.elements.lon.data, self.sim.elements.lat.data
        lon_deactive, lat_deactive = self.sim.elements_deactivated.lon.data, self.sim.elements_deactivated.lat.data
        
        try:
            init_lons_, init_lats_, lon_active_, lat_active_, lon_deactive_, lat_deactive_ = load_sim_results(outfile)

            init_lons = np.concatenate((init_lons, init_lons_), axis=0)
            init_lats = np.concatenate((init_lats, init_lats_), axis=0)
            lon_active = np.concatenate((lon_active, lon_active_), axis=0)
            lat_active = np.concatenate((lat_active, lat_active_), axis=0)
            lon_deactive = np.concatenate((lon_deactive, lon_deactive_), axis=0)
            lat_deactive = np.concatenate((lat_deactive, lat_deactive_), axis=0)
            print(f'\nLoaded existing file and combined it with the generated data.\n')
        except: 
            print(f'\nCould not load existing data from file.\n')


        np.savez_compressed(outfile, 
                            init_lons=init_lons, init_lats=init_lats, 
                            lon_active=lon_active, lat_active=lat_active,
                            lon_deactive=lon_deactive, lat_deactive=lat_deactive)
        

if __name__ == '__main__':

    set_seed(seed=42) # Reproducibility

    # Run multiple times to reduce variance in sampled date
    for _ in range(10):
        simulator = GhostNetSimulator()
        #simulator.run(time_step=3600, steps=24*6, outfile=f'./simulation_results/ghostnet_points.npz')

        






    


    

    