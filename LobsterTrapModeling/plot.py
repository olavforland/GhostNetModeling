import plotly.express as px
import plotly.graph_objects as go

CENTER = {'lat': 64, 'lon': 8}

def plot_geometry(gdf,
                  color=None,
                  labels=None,
                  locations='index',
                  geometry='geometry',
                  mapbox_style="open-street-map",
                  hover_data=None,
                  zoom=9.5,
                  opacity=0.6,
                  height=1000,
                  width=1400,
                  show=True,
                  color_continuous_scale='rdylgn'):
    """Helper for plotting choropleth geometries. 
    """
    
    # Retrieve index
    if isinstance(locations, str):
        locations = gdf.index if locations == 'index' else gdf[locations]

    # Retrieve geometry
    if isinstance(geometry, str):
        geometry = gdf[geometry]

    fig = px.choropleth_mapbox(gdf,
                               geojson=geometry,
                               color=color,
                               labels=labels,
                               locations=locations,
                               center=CENTER,
                               hover_data=hover_data,
                               mapbox_style=mapbox_style,
                               zoom=zoom,
                               opacity=opacity,
                               color_continuous_scale=color_continuous_scale)

    fig.update_layout(height=height, width=width, margin={"r": 0, "t": 0, "l": 0, "b": 0}, )
    if show:
        fig.show()
    return fig

def scatter_trace(df, lat, lon, mode='markers', size=8, color='blue', text=''):
    """Helper for generating a trace of scatterpoints.
    """
    if isinstance(lat, str):
        lat = df[lat]
    if isinstance(lon, str):
        lon = df[lon]

    scatter_trace = go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode=mode,
        marker=go.scattermapbox.Marker(
            size=size, 
            color=color
        ),
        text=text
    )

    return scatter_trace

def plot_scatter(df, lat, lon, color, labels, size=8, mode='markers', zoom=8, height=1000, width=1400, show=True):
    
    fig = go.Figure()
    found_trace = scatter_trace(df, lat, lon, color=color, text=labels, size=size)
    fig.add_trace(found_trace)
    fig.update_layout(mapbox_style='open-street-map', margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=height, width=width)
    
    if show:
        fig.show()
    return fig