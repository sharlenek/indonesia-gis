import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import ListedColormap

# --- CONFIG ---
BUFFER_KM = 10  # Change to match your data file
YEAR_TO_PLOT = 2021

# Paths
results_csv = f"../../Tables/Palm Analysis/Buffer{BUFFER_KM}km.csv"
village_csv = "../../Raw/village_locations.csv"
tile_shapefile = "../../Raw/Grid_OilPalm2016-2021.shp"

# --- Load data ---
df = pd.read_csv(results_csv)
village_df = pd.read_csv(village_csv)
tiles_gdf = gpd.read_file(tile_shapefile)

# --- Filter results for selected year ---
df_year = df[df['Year'] == YEAR_TO_PLOT]

# --- Merge village locations ---
merged = pd.merge(df_year, village_df, on='VillageID', how='left')
merged['geometry'] = merged.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")

# --- Plot ---
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Plot palm tile footprints as transparent outlines
tiles_gdf.to_crs("EPSG:4326").plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5)

# Plot villages: skipped = gray, valid = green
status_cmap = ListedColormap(['lightgray', '#1b9e77'])
gdf.plot(ax=ax, column='Status', legend=True, markersize=10, cmap=status_cmap)

ax.set_title(f"Village Status with Palm Tile Footprints — Buffer {BUFFER_KM}km — {YEAR_TO_PLOT}")
ax.set_axis_off()
plt.tight_layout()
plt.show()
