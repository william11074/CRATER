import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import folium
from folium.plugins import HeatMap
from shapely.geometry import MultiPoint
import datetime
import webbrowser

class PotholeRepairOptimizer:
    def __init__(self, filepath="pothole_dataset.csv"):
        self.df = pd.read_csv(filepath)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        self.work_orders_filename = "work_orders.html"
    
    def visualize_routes(self, show_polygons=True, show_heatmap=True):
        """Create interactive map with optimized routes and heatmap layer"""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")
            
        center = self.df[['latitude', 'longitude']].mean().values.tolist()
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles='Stadia.AlidadeSmoothDark',
            attr=' '  # This sets the attribution to a single space
        )
        
        # ========== HEATMAP LAYER ==========
        if show_heatmap:
            # Generate heatmap data: list of [lat, lon] points
            heat_data = [[row['latitude'], row['longitude']] for _, row in self.df.iterrows()]
            
            # Create heatmap with custom gradient
            heatmap = HeatMap(
                heat_data,
                name='Pothole Density',
                min_opacity=0.3,
                max_zoom=16,
                radius=30,
                blur=30,
                gradient={
                    0.1: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime',
                    0.7: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)
        
        # Add noise points (outliers)
        noise_points = self.df[self.df['cluster'] == -1]
        for _, row in noise_points.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='#555555',
                fill=True,
                fill_color='#555555',
                fill_opacity=0.7,
                popup="Isolated pothole"
            ).add_to(noise_marker_layer)

        # Add cluster polygons if enabled
        if show_polygons:
            for cluster_id, poly in self.cluster_polygons.items():
                try:
                    bounds = [[p[0], p[1]] for p in poly.exterior.coords]
                    folium.Polygon(
                        locations=bounds,
                        color='#555555',
                        fill=True,
                        fill_color='#555555',
                        fill_opacity=0.1,
                        weight=1,
                        popup=f"Cluster {cluster_id}"
                    ).add_to(polygon_group)
                except:
                    pass
        
        folium.LayerControl().add_to(m)
        return m
    
    def generate_work_orders(self):
        """Generate work orders with route assignments"""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")
            
        work_orders = []
        for cluster_id, route_df in self.cluster_routes.items():
            route_df = route_df.copy()
            route_df['route_id'] = cluster_id
            route_df['crew_assignment'] = f"Crew {cluster_id % 3 + 1}"  # Assign to 3 crews
            work_orders.append(route_df)
            
        return pd.concat(work_orders).sort_values(['route_id', 'route_order'])

# Example usage
if __name__ == "__main__":
    optimizer = PotholeRepairOptimizer("pothole_dataset.csv")
    
    # Step 1: Cluster potholes
    clustered_data = optimizer.cluster_potholes()
    print(f"Found {len(clustered_data['cluster'].unique())} clusters")
    
    # Step 2: Optimize routes
    routes, stats = optimizer.optimize_routes()
    print("\nRoute Statistics:")
    for cluster_id, stat in stats.items():
        print(f"Cluster {cluster_id}: {stat['num_potholes']} potholes, "
              f"{stat['total_km']}km, {stat['total_time_mins']}mins")
    
    # Step 3: Generate work orders HTML first
    html_file = optimizer.generate_work_orders_html('work_orders.html')
    print(f"\nWork orders saved to {html_file}")
    
    # Step 4: Visualize with heatmap and polygon toggle
    map = optimizer.visualize_routes(show_polygons=True, show_heatmap=True)
    map.save('pothole_repair_routes.html')
    print("\nRoute visualization saved to pothole_repair_routes.html")
    
    # Open the map in default browser
    webbrowser.open('pothole_repair_routes.html')
