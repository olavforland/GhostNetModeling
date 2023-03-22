import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

from plotly.colors import sample_colorscale
from joblib import dump, load

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans

from global_land_mask import globe

from helpers import compute_convex_hull, confidence_ellipse
from plot import plot_geometry, scatter_trace, plot_scatter

from typing import Tuple, List

# Some points that weren't caught by dbscan
# These are located in places that either doesn't make sense (in the middle of the country),
# or places which are too hard to reach (too far out in the ocean)
LOST_POINTS_TO_REMOVE = [
    (58.487226, 6.711539),
    (58.448028, 6.496214),
    (58.185317, 7.630616),
    (57.741315, 7.639418),
    (58.467932, 6.080887),
    (64.540024, 18.105448),
    (58.352757, 6.901853), 
    (58.472724, 9.482728)
]

FOUND_POINTS_TO_REMOVE = [
    (58.472724, 9.482728)
]

# Ignore pandas warnings on slices
pd.options.mode.chained_assignment = None  # default='warn'



class LobsterTrapLocationModel():
    """ Class for modelling both found and lost lobster traps. 
    """
    def __init__(self, data: pd.DataFrame=None, model: str='GMM', n_components: int=50, covariance_type: str='full', max_iter: int=500, 
                 weight_concentration_prior: float=None, weight_concentration_prior_type: str='dirichlet_process') -> None:
        """Init the class from data. 

        Args:
            data (pd.DataFrame, optional): Either data over lost or found lobster traps. Must contain columns `latitude`, `longitude`. Defaults to None.
            model (str, optional): Inference model. Either GMM (Gaussian Mixture Model) and BGMM (Bayesian Gaussian Mixture Model). Defaults to 'BGMM'.
            n_components (int, optional): See Sci-Kit Learn docs. Defaults to 20.
            covariance_type (str, optional): See Sci-Kit Learn docs. Defaults to 'full'.
            max_iter (int, optional): See Sci-Kit Learn docs. Defaults to 500.
            weight_concentration_prior (float, optional): See Sci-Kit Learn docs. Defaults to None.
            weight_concentration_prior_type (str, optional): See Sci-Kit Learn docs. Defaults to 'dirichlet_process'.
        """
        self.init_model(
            model, n_components, covariance_type, max_iter, weight_concentration_prior=weight_concentration_prior, 
            weight_concentration_prior_type=weight_concentration_prior_type
        )
        self.data = data
        self.n_components = n_components
        self.efficient_components = list(range(n_components))

        self.means_ = None
        self.covariances_ = None



    def init_model(self,  model: str='GMM', n_components: int=50, covariance_type: str='full', max_iter: int=500, 
                   weight_concentration_prior: float=None, weight_concentration_prior_type: str='dirichlet_process') -> None:
        """ Helper for (re-)initializing the model.
        """
        if model == 'GMM':
            self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter)
        elif model == 'BGMM':
            self.model = BayesianGaussianMixture(
                n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, weight_concentration_prior=weight_concentration_prior,
                weight_concentration_prior_type=weight_concentration_prior_type)
        else: 
            raise ValueError(f'Model type {model} not supported. Must be either GMM (Gaussian Mixture Model) or BGMM (Bayesian Gaussian Mixture Model)')


    def fit(self) -> "LobsterTrapLocationModel":
        """Fits the data provided when initializing.

        Returns:
            LobsterTrapLocationModel: The fitted model
        """
        X = self.data[['latitude', 'longitude']]
        self.model = self.model.fit(X)
        self.means_ = self.model.means_
        self.covariances_ = self.model.covariances_

        if isinstance(self.model, BayesianGaussianMixture):
            self.efficient_components = np.unique(self.predict())
        return self
    
    def predict(self) -> np.ndarray:
        """Predict the labels of the fitted data.

        Returns:
            np.ndarray: The predicted labels.
        """
        X = self.data[['latitude', 'longitude']]
        self.data['label'] = self.model.predict(X)
        return self.data['label'].values
    

    def detect_outliers_dbscan(self,  eps: float=0.5, min_samples: int=5, plot_outliers: bool=False, text: str=None, color_labels: bool=False) -> np.ndarray:
        """Detect outliers using DBSCAN algorithm. Outliers are points not classified by the algorithm (label=-1).

        Args:
            eps (float, optional): See Sci-Kit Learn docs. Defaults to 0.5.
            min_samples (int, optional): See Sci-Kit Learn docs. Defaults to 5.
            plot_outliers (bool, optional): Whether to plot all points and mark the outliers. Defaults to False.
            text (str, optional): Marker text for plotting. Defaults to None.

        Returns:
            np.ndarray: Binary outlier mask for the data points.
        """
        X = self.data[['latitude', 'longitude']]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        # Outliers are detected as noise in the data, get label -1
        outlier_mask = labels == -1
        color = outlier_mask if color_labels else labels
        if plot_outliers:
            if text:
                text = self.data[text]
            fig = px.scatter_mapbox(self.data, lat='latitude', lon='longitude', mapbox_style='open-street-map', color=color, hover_name=text)
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=1000, width=1400)
            fig.show()

        return outlier_mask



    def remove_outliers_dbscan(self, eps: float, min_samples: int, plot_outliers: bool, 
                               manual_points_to_remove: List[Tuple[float]]) -> pd.DataFrame:
        """Detects outliers and removes them using the DBSCAN algorithm.

        Args:
            eps (float): See Sci-Kit Learn docs.
            min_samples (int): See Sci-Kit Learn docs.
            plot_outliers (bool): Whether to plot all points and mark the outliers.
            manual_points_to_remove (Iterable[Tuple): Specify manual points to remove as well.

        Returns:
            pd.DataFrame: The data after removing outliers.
        """
        outlier_mask = self.detect_outliers_dbscan(eps=eps, min_samples=min_samples, plot_outliers=plot_outliers)
        coords = self.data.apply(lambda x: (np.round(x['latitude'], 6), np.round(x['longitude'], 6)), axis=1)

        # Manually remove outlier that isn't picket up by DBSCAN
        outlier_mask |= coords.isin(manual_points_to_remove)

        print(f'Successfully removed {outlier_mask.sum()} outliers ({outlier_mask.mean() * 100:.2f}%)')
        self.data = self.data[~outlier_mask]
        return self.data
    
    def silhouette_score(self, X: np.ndarray=None, labels: np.ndarray=None) -> float:
        """Calculates the silhouette score for the given data, or the data in the model if not provided.

        Args:
            X (np.ndarray, optional): Latitude and longitude coordinates. Defaults to None.
            labels (np.ndarray, optional): Predicted labels corresponding to each coordinate. Defaults to None.

        Returns:
            float: Silhouette score.
        """
        if X is None:
            X = self.data[['latitude', 'longitude']]
        if labels is None: 
            labels = self.data['label']

        score = silhouette_score(X, labels)
        return score
    

    def plot(self, include_conf_ellipse: bool=True, conf_level: float=0.95, include_hull: bool=False, sample_hull: bool=False, num_samples: int=100000) -> go.Figure:
        """Plot the data in the model. 

        Args:
            include_conf_ellipse (bool, optional): Whether to include the conf-level ellipses for each the mixture models. Defaults to True.
            conf_level (float, optional): Confidence level for the ellipses. Determines the size of the ellipses. Defaults to 0.95.
            include_hull (bool, optional): Whether to plot the connvex hull around each mixture (Not in use). Defaults to False.
            sample_hull (bool, optional): Whether to calculate the convex hull around sampled points or the actual points (Not in use). Defaults to False.
            num_samples (int, optional): How many points to sample when using sampled points to get convex hull. Defaults to 100000.

        Returns:
            go.Figure: The resulting figure.
        """
        scatter_color = self.data['label']

        if include_hull:
            if sample_hull:
                hull = compute_convex_hull(self.data, label='label', dist=self.model, num_samples=num_samples)
            else:
                hull = compute_convex_hull(self.data, label='label')
            fig = plot_geometry(hull, geometry='conv_hull', zoom=3.5, color='label', show=False)
        
        elif include_conf_ellipse:
            fig = go.Figure()

            colors = sample_colorscale('rainbow', self.n_components)

            for i in self.efficient_components:
                color = colors[i]
                x, y = confidence_ellipse(mean=self.model.means_[i], cov=self.model.covariances_[i], conf_level=conf_level)
                fig.add_scattermapbox(
                            lat=x,
                            lon=y,
                            mode='lines',
                            fill='toself', 
                            line=go.scattermapbox.Line(width=3, color=color)
                )

            scatter_color = self.data['label'].apply(lambda x: colors[x])
        else:
            fig = go.Figure()
        found_trace = scatter_trace(self.data, 'latitude', 'longitude', color=scatter_color, text=self.data['label'])
        fig.add_trace(found_trace)
        
        fig.update_layout(mapbox_style='open-street-map', margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=1000, width=1400)
        fig.show()


    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculates the likelihood of belonging to the model for each point in X.

        Args:
            X (np.ndarray): The coordinates to score

        Returns:
            np.ndarray: The resulting likelihoods
        """
        return np.exp(self.model.score_samples(X))
    
    def dump(self, filename: str) -> None:
        """Save the model to file.

        Args:
            filename (str): Name of / path to file to save model to.
        """
        dump(self.state_dict(), filename)
        print(f'Successfully saved model at {filename}')

    def load(self, filename: str) -> None:
        """Load model from file.

        Args:
            filename (str): Name of / path to file to save model to.
        """
        state_dict = load(filename)
        self.model = state_dict['model']
        self.data = state_dict['data']
        self.n_components = state_dict['n_components']
        self.efficient_components = state_dict['efficient_components']
        self.means_ = state_dict['means_']
        self.covariances_ = state_dict['covariances_']

    @staticmethod
    def from_dict(state_dict):
        model = LobsterTrapLocationModel()
        model.model = state_dict['model']
        model.data = state_dict['data']
        model.n_components = state_dict['n_components']
        model.efficient_components = state_dict['efficient_components']
        model.means_ = state_dict['means_']
        model.covariances_ = state_dict['covariances_']
        return model

    def state_dict(self):
        return {
            'model': self.model,
            'data': self.data,
            'n_components': self.n_components,
            'efficient_components': self.efficient_components, 
            'means_': self.means_,
            'covariances_': self.covariances_
        }
    

class RegionalLobsterTrapModel:
    def __init__(self, data: pd.DataFrame=None, model: str='GMM', max_iter: int=500) -> None:
        self.data = data
        self.model_type = model
        
        self.labels = np.zeros(data.shape[0], dtype=int) if data is not None else None
        self.models = {}

    def cluster_data(self, n_clusters=7, plot=False) -> np.ndarray:
        
        clusters = KMeans(n_clusters=n_clusters, n_init='auto').fit(self.data[['latitude', 'longitude']])    
        self.labels = clusters.predict(self.data[['latitude', 'longitude']])
        if plot:
            _ = plot_scatter(self.data, 'latitude', 'longitude', color=self.labels, labels=self.labels)
        return self.labels
    
    def find_best_num_clusters(self, n_clusters_space=range(6, 15), scoring='silhouette', verbose=False):
        # Dictionary from n_clusters to mean score of every fitted model through every cluster
        n_clusters_mean_scores = {}
        # Dictionary from n_clusters to a dictionary containing number of Gaussian components within each cluster
        n_clusters_to_components_within_clusters = {}

        # Try different clusters for detecting areas
        for n_clusters in n_clusters_space: 
            if verbose:
                print(f'\n--------- Number of clusters = {n_clusters} ---------\n')

            clusters = KMeans(n_clusters=n_clusters, n_init='auto').fit(self.data[['latitude', 'longitude']])
            labels = clusters.predict(self.data[['latitude', 'longitude']])

            cluster_scores = []
            components_within_cluster = {}

            # Go through each of the clusters
            for label in np.unique(labels):
                best_components_score = (-1, -1) # (n_components, score)
                
                # Num samples in the current cluster
                num_samples = (labels == label).sum()

                # Try all legal components that are no larger than 7 in the current cluster 
                for n_components in range(2, min(num_samples-1, 7)):

                    scores = []
                    # Fit model 10 times to reduce variance in EM algorithm
                    for _ in range(10):

                        model = LobsterTrapLocationModel(data=self.data[labels == label], model='GMM', n_components=n_components)

                        model.fit()
                        model.predict()
                        if scoring == 'silhouette':
                            score = model.silhouette_score()
                        else:
                            pass #TODO
                        scores.append(score)
                    # Take mean of 5 best scores
                    score = np.mean(sorted(scores)[-5:])
                    if score > best_components_score[1]:
                        best_components_score = (n_components, score)

                best_components, best_score = best_components_score
                if verbose:
                    print(f'  ----> Cluster {label}: [n_components={best_components}, silhouette_score={best_score}]')

                cluster_scores.append(best_score)
                components_within_cluster[label] = best_components


            n_clusters_mean_scores[n_clusters] = np.mean(cluster_scores)
            n_clusters_to_components_within_clusters[n_clusters] = components_within_cluster
            if verbose:
                print(f'\n--------- Fitted GMM to every clusters with mean silhouette score = {np.mean(cluster_scores):.3f} -----------')
                print('-------------------------------------------------------------------------------------------------------')

        return n_clusters_mean_scores, n_clusters_to_components_within_clusters
    

    def fit_trap_models(self, labels_2_n_components, scoring='silhouette', verbose=False):
        models = {} # label to LobsterTrapLocationModel
        for label in np.unique(self.labels):
            best_components_silhouette = (-1, -1) # (n_components, silhouette)
            best_model = None

            n_components = labels_2_n_components[label]
            # Run 5 times and pick best GMM fit
            for _ in range(5):

                model = LobsterTrapLocationModel(data=self.data[self.labels == label], model='GMM', n_components=n_components)

                model.fit()
                model.predict()
                if scoring == 'silhouette':
                    score = model.silhouette_score()
                else: 
                    pass #TODO

                if score > best_components_silhouette[1]:
                    best_components_silhouette = (n_components, score)
                    best_model = model

            models[label] = best_model
            best_components, best_score = best_components_silhouette
            if verbose:
                print(f'---> GMM for cluster {label}: [n_components={best_components}, silhouette_score={best_score}]')
        self.models = models
        return models
    

    def plot(self, conf_level=0.95):
        fig = go.Figure()

        colors = sample_colorscale('rainbow', max(len(np.unique(self.labels)), 2))

        for label in np.unique(self.labels):
            label_model = self.models[label]
            label_data = self.data[self.labels == label]
            label_color = colors[label]

            text = [f'Cluster: {l}' for l in self.labels[self.labels == label]]
            # Go thgough every component
            for i in range(len(label_model.means_)):
                color = colors[label]
                x, y = confidence_ellipse(mean=label_model.means_[i], cov=label_model.covariances_[i], conf_level=conf_level)
                fig.add_scattermapbox(
                            lat=x,
                            lon=y,
                            mode='lines',
                            fill='toself', 
                            line=go.scattermapbox.Line(width=3, color=color),
                            text=text
                )

            found_trace = scatter_trace(label_data, 'latitude', 'longitude', color=label_color, text=text)
            fig.add_trace(found_trace)
        
        fig.update_layout(mapbox_style='open-street-map', margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=1000, width=1400)
        fig.show()



    def silhouette_score(self):
        return np.mean([model.silhouette_score() for model in self.models.values()])



    def change_cluster_components(self, cluster_num, new_components, plot=True):
        best_components_silhouette = (-1, -1)
        for _ in range(5):

            model = LobsterTrapLocationModel(data=self.data[self.labels == cluster_num], model='GMM', n_components=new_components)

            model.fit()
            model.predict()
            score = model.silhouette_score()

            if score > best_components_silhouette[1]:
                best_components_silhouette = (new_components, score)
                best_model = model

        self.models[cluster_num] = best_model
        if plot:
            self.plot()
        return best_model
    

    def detect_outliers_dbscan(self,  eps: float=0.5, min_samples: int=5, plot_outliers: bool=False, text: str=None, color_labels: bool=False, manual_outliers=None) -> np.ndarray:
        """Detect outliers using DBSCAN algorithm. Outliers are points not classified by the algorithm (label=-1).

        Args:
            eps (float, optional): See Sci-Kit Learn docs. Defaults to 0.5.
            min_samples (int, optional): See Sci-Kit Learn docs. Defaults to 5.
            plot_outliers (bool, optional): Whether to plot all points and mark the outliers. Defaults to False.
            text (str, optional): Marker text for plotting. Defaults to None.

        Returns:
            np.ndarray: Binary outlier mask for the data points.
        """
        X = self.data[['latitude', 'longitude']]
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        # Outliers are detected as noise in the data, get label -1
        outlier_mask = labels == -1
        # Enter manual points
        coords = X.apply(lambda x: (np.round(x['latitude'], 6), np.round(x['longitude'], 6)), axis=1)
        outlier_mask |= coords.isin(manual_outliers)
        color = outlier_mask if not color_labels else labels
        if plot_outliers:
            if text:
                text = self.data[text]
            fig = px.scatter_mapbox(self.data, lat='latitude', lon='longitude', mapbox_style='open-street-map', color=color, hover_name=text)
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=1000, width=1400)
            fig.show()

        return outlier_mask



    def remove_outliers_dbscan(self, eps: float, min_samples: int, plot_outliers: bool, 
                               manual_points_to_remove: List[Tuple[float]]) -> pd.DataFrame:
        """Detects outliers and removes them using the DBSCAN algorithm.

        Args:
            eps (float): See Sci-Kit Learn docs.
            min_samples (int): See Sci-Kit Learn docs.
            plot_outliers (bool): Whether to plot all points and mark the outliers.
            manual_points_to_remove (Iterable[Tuple]): Specify manual points to remove as well.

        Returns:
            pd.DataFrame: The data after removing outliers.
        """
        outlier_mask = self.detect_outliers_dbscan(eps=eps, min_samples=min_samples, plot_outliers=plot_outliers, manual_outliers=manual_points_to_remove)
        coords = self.data.apply(lambda x: (np.round(x['latitude'], 6), np.round(x['longitude'], 6)), axis=1)

        # Manually remove outlier that isn't picket up by DBSCAN
        outlier_mask |= coords.isin(manual_points_to_remove)

        print(f'Successfully removed {outlier_mask.sum()} outliers ({outlier_mask.mean() * 100:.2f}%)')
        self.data = self.data[~outlier_mask]
        self.labels = np.zeros(self.data.shape[0], dtype=int)
        return self.data
    
    def state_dict(self):
        return {
            'models': {label: model.state_dict() for label, model in self.models.items()},
            'data': self.data,
            'labels': self.labels,
            'model_type': self.model_type
        }
        
    
    def dump(self, filepath: str) -> None:
        """Save the model to file.

        Args:
            filename (str): Name of / path to file to save model to.
        """
        # to_dump = (len(self.models), ) + tuple(model.to_dict() for model in self.models) + (
        #     self.data, 
        #     self.labels,
        #     self.model_type
        # )
        dump(self.state_dict(), filepath)
        print(f'Successfully saved model at {filepath}')

    def load(self, filepath: str) -> None:
        """Load model from file.

        Args:
            filename (str): Name of / path to file to save model to.
        """
        state_dict = load(filepath)
        self.models = {
            label: LobsterTrapLocationModel.from_dict(state_dict) 
            for label, state_dict in state_dict['models'].items()
        }
        self.data = state_dict['data']
        self.labels = state_dict['labels']
        self.model_type = state_dict['model_type']
        
    
    def score_samples(self, points): 

        scores = np.array([model.score_samples(points) for model in self.models.values()])
        scores = scores.clip(0.001, 0.9)
        scores = np.sum(scores, axis=0)
        scores = scores / scores.max()

        #print(scores.shape)
        return scores


class RegionalFoundTrapsLocationModel(RegionalLobsterTrapModel):

    def detect_outliers_dbscan(self,  eps: float=0.5, min_samples: int=8, plot_outliers: bool=False, text: str=None, color_labels: bool=False, manual_outliers=FOUND_POINTS_TO_REMOVE) -> np.ndarray:
        return super().detect_outliers_dbscan(eps, min_samples, plot_outliers, text, color_labels, manual_outliers=manual_outliers)
    
    def remove_outliers_dbscan(self, eps=0.501, min_samples=8, plot_outliers=False,
                                     manual_points_to_remove=FOUND_POINTS_TO_REMOVE) -> pd.DataFrame:
        return super().remove_outliers_dbscan(eps, min_samples, plot_outliers, manual_points_to_remove)

class RegionalLostTrapsLocationModel(RegionalLobsterTrapModel):

    def detect_outliers_dbscan(self, eps: float=0.25, min_samples: int=40, plot_outliers: bool=False, text: str=None, color_labels: bool=False, manual_outliers=LOST_POINTS_TO_REMOVE) -> np.ndarray:
        # Some outliers not picked up by DBSCAN algorithm
        is_ocean = globe.is_ocean(self.data['latitude'], self.data['longitude'])
        self.data = self.data[~((self.data['num_tools_lost'] > 25) & ~is_ocean)]
        return super().detect_outliers_dbscan(eps, min_samples, plot_outliers, text, color_labels, manual_outliers=manual_outliers)
    
    def remove_outliers_dbscan(self, eps: float=0.25, min_samples: int=40, plot_outliers: bool=False, 
                                    manual_points_to_remove=LOST_POINTS_TO_REMOVE) -> pd.DataFrame:
        # Some outliers not picked up by DBSCAN algorithm
        is_ocean = globe.is_ocean(self.data['latitude'], self.data['longitude'])
        self.data = self.data[~((self.data['num_tools_lost'] > 25) & ~is_ocean)]

        return super().remove_outliers_dbscan(eps, min_samples, plot_outliers, manual_points_to_remove)


            


                
                
