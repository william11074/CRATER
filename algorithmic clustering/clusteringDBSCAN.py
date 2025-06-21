import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import folium
from shapely.geometry import MultiPoint
import datetime

class PotholeRepairOptimizer:
    def __init__(self, filepath="pothole_dataset.csv"):
        self.df = pd.read_csv(filepath)
        self.cluster_routes = {}
        self.route_stats = {}
        self.cluster_polygons = {}
        
    def cluster_potholes(self, eps_km=0.2, min_samples=3):
        """Cluster potholes using DBSCAN with haversine distance"""
        coords = self.df[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        kms_per_radian = 6371.0088
        epsilon = eps_km / kms_per_radian
        
        db = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            algorithm='ball_tree',
            metric='haversine'
        ).fit(coords_rad)
        
        self.df['cluster'] = db.labels_
        
        # Create cluster polygons
        for cluster_id in set(db.labels_):
            if cluster_id == -1:
                continue
            cluster_points = self.df[self.df['cluster'] == cluster_id][['latitude', 'longitude']].values
            if len(cluster_points) >= 3:
                try:
                    poly = MultiPoint(cluster_points).convex_hull
                    self.cluster_polygons[cluster_id] = poly
                except:
                    pass
        
        return self.df
    
    def visualize_routes(self, show_polygons=True):
        """Create interactive map with optimized routes"""
        if not self.cluster_routes:
            raise ValueError("Run optimize_routes() first")
            
        center = self.df[['latitude', 'longitude']].mean().values.tolist()
        m = folium.Map(
            location=center,
            zoom_start=12,
            tiles='Stadia.AlidadeSmoothDark',
            attr=' '  # This sets the attribution to a single space
        )
        
        # Subtle color palette for routes
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Create layer groups
        route_lines = folium.FeatureGroup(name='Repair Routes', show=True).add_to(m)
        route_markers = folium.FeatureGroup(name='Route Markers', show=True).add_to(m)
        polygon_group = folium.FeatureGroup(name='Cluster Polygons', show=show_polygons).add_to(m)
        noise_marker_layer = folium.FeatureGroup(name='Isolated Potholes', show=True).add_to(m)
        
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
        
        # Add routes and markers
        for i, (cluster_id, route_df) in enumerate(self.cluster_routes.items()):
            color = colors[i % len(colors)]
            
            # Add route line
            folium.PolyLine(
                locations=route_df[['latitude', 'longitude']].values.tolist(),
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Route {cluster_id}<br>"
                      f"Potholes: {len(route_df)}<br>"
                      f"Distance: {self.route_stats[cluster_id]['total_km']}km<br>"
                      f"Time: {self.route_stats[cluster_id]['total_time_mins']}mins"
            ).add_to(route_lines)
            
            # Add markers with order numbers (smaller and subtle)
            for idx, row in route_df.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,  # Smaller markers
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Route {cluster_id}<br>"
                          f"Order: {row['route_order']+1}<br>"
                          f"Severity: {row.get('severity', 'N/A')}"
                ).add_to(route_markers)
        
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
    
    # Step 4: Visualize with polygon toggle
    map.save('pothole_repair_routes.html')
    print("\nRoute visualization saved to pothole_repair_routes.html")
