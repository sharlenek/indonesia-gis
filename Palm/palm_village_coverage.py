import os
import rasterio
import numpy as np
import pandas as pd
from shapely.geometry import box, Point
import geopandas as gpd
from rasterio.mask import mask
from rtree import index
import time

# --- CONFIG ---
BUFFER_KM = 10  # ← change this to 5, 10, 20, or 30 as needed
village_csv = "../../Raw/village_locations.csv"
palm_folder = "../../Raw/Palm Coverage/Filtered_Palm_Tiles"
water_folder = "../../Processed/Aligned_Water_Mask"
output_dir = "../../Tables/Palm Analysis"
final_csv = os.path.join(output_dir, f"Buffer{BUFFER_KM}km.csv")
years = list(range(1990, 2022))
pixel_area_ha = 0.09

# --- Load and buffer villages ---
vdf = pd.read_csv(village_csv)
geometry = [Point(xy) for xy in zip(vdf['lon'], vdf['lat'])]
gdf = gpd.GeoDataFrame(vdf, geometry=geometry, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=32748)
gdf[f'buffer_{BUFFER_KM}km'] = gdf.geometry.buffer(BUFFER_KM * 1000)
gdf = gdf.set_geometry(f'buffer_{BUFFER_KM}km').set_crs(epsg=32748).to_crs("EPSG:4326")

# --- Index palm tiles ---
palm_idx = index.Index()
tile_meta = {}

for i, fname in enumerate(sorted(os.listdir(palm_folder))):
    if not fname.endswith('.tif'):
        continue
    fpath = os.path.join(palm_folder, fname)
    with rasterio.open(fpath) as src:
        bounds = box(*src.bounds)
        palm_idx.insert(i, bounds.bounds)
        tile_meta[i] = {
            "palm": fpath,
            "water": os.path.join(water_folder, fname.replace(".tif", "_water_mask.tif")),
            "bounds": bounds
        }

# --- Process villages ---
os.makedirs(output_dir, exist_ok=True)
results = []
start = time.time()

for idx, village in gdf.iterrows():
    if idx % 100 == 0:
        print(f"🧭 Processing village {idx + 1}/{len(gdf)} — VillageID={village['VillageID']}")

    vid = village['VillageID']
    adm_code = village['ADM3_PCODE']
    adm_name = village['ADM3_EN']
    buffer_shape = village[f'buffer_{BUFFER_KM}km']
    buffer_geom = [buffer_shape]
    buffer_bounds = buffer_shape.bounds

    palm_stats = {y: {'new': 0, 'cum': 0} for y in years}
    total_land_pixels = 0
    matched = False

    for i in list(palm_idx.intersection(buffer_bounds)):
        meta = tile_meta[i]
        palm_path = meta["palm"]
        water_path = meta["water"]

        if not os.path.exists(water_path):
            continue

        if not meta["bounds"].intersects(buffer_geom[0]):
            continue

        matched = True

        try:
            with rasterio.open(palm_path) as palm_src, rasterio.open(water_path) as water_src:
                palm_clip, _ = mask(palm_src, buffer_geom, crop=True)
                water_clip, _ = mask(water_src, buffer_geom, crop=True)

                palm_data = palm_clip[0]
                water_data = water_clip[0]
                land_mask = (water_data != 1)

                if palm_data.size == 0 or land_mask.sum() == 0:
                    continue

                total_land_pixels += np.sum(land_mask)

                for year in years:
                    palm_year_mask = (palm_data == year)
                    palm_cum_mask = (palm_data <= year) & (palm_data >= 1990)
                    palm_stats[year]['new'] += np.sum(palm_year_mask & land_mask)
                    palm_stats[year]['cum'] += np.sum(palm_cum_mask & land_mask)

        except Exception as e:
            print(f"  ⚠️ Skipping tile {os.path.basename(palm_path)} due to error: {e}")
            continue

    for year in years:
        result = {
            'VillageID': vid,
            'ADM3_PCODE': adm_code,
            'ADM3_EN': adm_name,
            'Year': year
        }

        if not matched:
            result.update({
                'NewPalm_ha': np.nan,
                'CumulativePalm_ha': np.nan,
                'TotalLand_ha': np.nan,
                'PalmPercent': np.nan,
                'Status': "skipped"
            })
        else:
            new_ha = palm_stats[year]['new'] * pixel_area_ha
            cum_ha = palm_stats[year]['cum'] * pixel_area_ha
            total_ha = total_land_pixels * pixel_area_ha
            percent = (cum_ha / total_ha * 100) if total_ha > 0 else 0

            result.update({
                'NewPalm_ha': new_ha,
                'CumulativePalm_ha': cum_ha,
                'TotalLand_ha': total_ha,
                'PalmPercent': percent,
                'Status': "valid"
            })

        results.append(result)


# Final write
df = pd.DataFrame(results)
df['NetChangePalm_ha'] = df.groupby('VillageID')['CumulativePalm_ha'].diff().fillna(0)
df.loc[df['Status'] == 'skipped', 'NetChangePalm_ha'] = np.nan

# --- Reorder columns for cleaner CSV ---
desired_order = [
    'VillageID', 'ADM3_PCODE', 'ADM3_EN', 'Year',
    'NewPalm_ha', 'CumulativePalm_ha', 'NetChangePalm_ha',
    'TotalLand_ha', 'PalmPercent', 'Status'
]

# Reindex dataframe to desired column order
df = df[desired_order]

df.to_csv(final_csv, index=False)

# Report skipped
skipped = df[df['Status'] == 'skipped']['VillageID'].nunique()
total = gdf['VillageID'].nunique()

print(f"\n✅ Final results saved to: {final_csv}")
print(f"⏱️ Total runtime: {(time.time() - start)/60:.2f} minutes")
print(f"⚠️ Villages skipped: {skipped} out of {total} ({(skipped/total)*100:.1f}%)")
