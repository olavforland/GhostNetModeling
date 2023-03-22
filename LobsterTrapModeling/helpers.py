import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd

import plotly.express as px

from scipy.stats import chi2
from global_land_mask import globe
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN
from typing import Tuple


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def grid_lon_to_km(lat: float) -> float:
    """Calculates kilometres per degree longitude, at a certain latitude. The quantity shrinks when approaching the poles.
    """
    # Convert latitude to radians
    lat = lat * np.pi / 180

    # Calculate the length of 1 degree of longitude in km
    # at the given latitude using the WGS-84 ellipsoid
    a = 6378.137  # equatorial radius of the Earth in km
    b = 6356.7523  # polar radius of the Earth in km
    lon_km = 2 * np.pi * a * np.cos(lat) / 360
    lon_km_polar = 2 * np.pi * b * np.cos(lat) / 360
    return (lon_km + lon_km_polar) / 2

# See LobsterTrapLocationModel
def dbscan_outlier_removal(df, eps=0.501, min_samples=5, plot_outliers=False):
    X = df[['latitude', 'longitude']]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    # Outliers are detected as noise in the data, get label -1
    outlier_mask = labels == -1
    if plot_outliers:
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', mapbox_style='open-street-map', color=outlier_mask)
        fig.show()

    # Manually remove outlier that isn't picket up by DBSCAN
    outlier_mask |= (df['latitude'] == 58.472724233) & (df['longitude'] == 9.482728355)
    print(f'Successfully removed {outlier_mask.sum()} outliers ({outlier_mask.mean() * 100:.2f}%)')
    return df[~outlier_mask]

# Creates convex hull for the points within each label (not in use)
def compute_convex_hull(df, label='label', geometry=None, dist=None, num_samples=100000):
    # Create point-geometry for the current points in group
    if dist: # If distribution is provided, sample points from it
        samples, labels = dist.sample(num_samples)
        df = pd.DataFrame({'latitude': samples[:, 0], 'longitude': samples[:, 1], 'label': labels})
    if dist or geometry is None:
        df['geometry'] = df.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    # Convert all points in each label group to multipoint
    area_df = df.groupby(label, as_index=False)['geometry'].apply(lambda x: MultiPoint(x.values))
    area_df = gpd.GeoDataFrame(area_df, geometry='geometry', index=area_df.index)
    area_df.loc[:, 'conv_hull'] = area_df['geometry'].convex_hull
    return gpd.GeoDataFrame(area_df[[label, 'conv_hull']], geometry='conv_hull')


def calculate_ocean_fraction(polygon, delta=0.1):
    """Estimates the fraction of ocean in a polygon (of coordinates)
    """
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    # Get number of points needed to get 10m x 10m accuracy
    num_lat_points = int(111 * (max_lat - min_lat) / delta)
    num_lon_points = int(grid_lon_to_km((max_lat + min_lat) / 2) * (max_lon - min_lon) / delta)

    # Create a vector of points over each axis in grid
    lats = np.linspace(min_lat, max_lat, num_lat_points)
    lons = np.linspace(min_lon, max_lon, num_lon_points)

    # Make grid, and transform to rows of points
    lon_grid, lat_grid = np.meshgrid(lons,lats)
    points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    
    # Create mask for points in polygon (time-consuming)
    in_polygon_mask = np.asarray([polygon.contains(Point(x, y)) for x, y in points])

    # Filter 
    points = points[in_polygon_mask]

    # Get whether the points is in the ocean using globe.is_ocean
    globe_land_mask = globe.is_ocean(points[:, 1], points[:, 0])

    return globe_land_mask.mean()



def confidence_ellipse(mean: np.ndarray, cov: np.ndarray, N: int=100, conf_level: float=0.95) -> Tuple[np.ndarray]:
    """ Calculates the confidence ellipse for mean-vector and covariance matrix at given confidence level.
    """

    # Eigenvalues and vectors
    x_mean, y_mean = mean
    eig_val, eig_vec = np.linalg.eigh(cov)

    # Scale
    alpha = chi2.ppf(conf_level, 2)
    a = np.sqrt(alpha * eig_val[0])
    b = np.sqrt(alpha * eig_val[1])

    t = np.linspace(0, 2*np.pi, N)
    # Ellipse parameterization
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    # Rotation matrix
    R = np.array([eig_vec[0], eig_vec[1]]).T
    # Coordinate of the ellipse points with respect to the system of axes
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_mean 
    y = yp + y_mean
    return x, y