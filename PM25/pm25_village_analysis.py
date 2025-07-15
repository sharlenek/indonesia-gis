import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pathlib import Path

def process_pm25(
    village_csv_path: Path,
    nc_folder_path: Path,
    output_dir: Path,
    figures_dir: Path,
    k_values: list = [3, 5, 10],
    years: list = list(range(1998, 2023)),
    is_monthly: bool = False
):
    # Load village data
    village_df = pd.read_csv(village_csv_path)
    village_coords = village_df[['lon', 'lat']].to_numpy()
    village_ids = village_df['VillageID'].to_numpy()
    village_codes = village_df['ADM3_PCODE'].to_numpy() if 'ADM3_PCODE' in village_df.columns else [None]*len(village_ids)
    village_names = village_df['ADM3_EN'].to_numpy() if 'ADM3_EN' in village_df.columns else [None]*len(village_ids)

    # Load one sample NetCDF file to get lat/lon grid
    if is_monthly:
        subdirs = sorted(nc_folder_path.glob("*/"))
        if not subdirs:
            raise FileNotFoundError(f"No subfolders found in: {nc_folder_path}")
        sample_files = sorted(subdirs[0].glob("*.nc"))
        if not sample_files:
            raise FileNotFoundError(f"No .nc files found in: {subdirs[0]}")
        sample_nc_file = sample_files[0]
    else:
        sample_files = sorted(nc_folder_path.glob("*.nc"))
        if not sample_files:
            raise FileNotFoundError(f"No .nc files found in: {nc_folder_path}")
        sample_nc_file = sample_files[0]

    ds_sample = xr.open_dataset(sample_nc_file)
    lats = ds_sample['lat'].values
    lons = ds_sample['lon'].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pixel_coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    tree = cKDTree(pixel_coords)

    results_by_k = {k: [] for k in k_values}

    for year in years:
        if is_monthly:
            year_folder = nc_folder_path / str(year)
            nc_files = sorted(year_folder.glob("*.nc"))
        else:
            nc_files = list(nc_folder_path.glob(f"*{year}*.nc"))

        if not nc_files:
            print(f"NetCDF for year {year} not found.")
            continue

        if is_monthly:
            for nc_file in nc_files:
                ds = xr.open_dataset(nc_file)
                pm25_array = ds['PM25'].values

                if pm25_array.ndim != 2:
                    print(f"Expected 2D monthly array in {nc_file.name}, got shape {pm25_array.shape}")
                    continue

                try:
                    month_str = nc_file.stem.split(".")[-1][:6]
                    month = int(month_str[-2:])
                except:
                    month = 1

                pm25_flat = pm25_array.ravel()

                for i, village in enumerate(village_coords):
                    dists, idxs_all = tree.query(village, k=max(k_values))
                    for k in k_values:
                        idxs = np.atleast_1d(idxs_all[:k])
                        avg_pm25 = np.nanmean(pm25_flat[idxs])
                        results_by_k[k].append({
                            "VillageID": village_ids[i],
                            "ADM3_PCODE": village_codes[i],
                            "ADM3_EN": village_names[i],
                            "Year": year,
                            "Month": month,
                            "Avg_PM25": avg_pm25
                        })
                ds.close()
        else:
            ds = xr.open_dataset(nc_files[0])
            pm25_array = ds['PM25'].values

            if pm25_array.ndim == 3:
                pm25_annual = np.nanmean(pm25_array, axis=0)
            else:
                pm25_annual = pm25_array

            pm25_flat = pm25_annual.ravel()

            for i, village in enumerate(village_coords):
                dists, idxs_all = tree.query(village, k=max(k_values))
                for k in k_values:
                    idxs = np.atleast_1d(idxs_all[:k])
                    avg_pm25 = np.nanmean(pm25_flat[idxs])
                    results_by_k[k].append({
                        "VillageID": village_ids[i],
                        "ADM3_PCODE": village_codes[i],
                        "ADM3_EN": village_names[i],
                        "Year": year,
                        "Avg_PM25": avg_pm25
                    })
            ds.close()

    for k in k_values:
        df_k = pd.DataFrame(results_by_k[k])
        suffix = "monthly" if is_monthly else "annual"
        index_cols = ["VillageID", "ADM3_PCODE", "ADM3_EN"]
        if is_monthly:
            df_pivot = df_k.pivot_table(index=index_cols, columns=["Year", "Month"], values="Avg_PM25")
            df_pivot.columns = [f"{y}_{m:02d}" for y, m in df_pivot.columns]
        else:
            df_pivot = df_k.pivot_table(index=index_cols, columns="Year", values="Avg_PM25")
            df_pivot.columns = [str(col) for col in df_pivot.columns]
        df_pivot.reset_index().to_csv(output_dir / f"pm25_{suffix}_{k}_nearest.csv", index=False)

    # === Map of 2022 only ===
    latest_df = pd.DataFrame([r for r in results_by_k[3] if r['Year'] == 2022])
    merged = latest_df.merge(village_df, on='VillageID')
    geometry = [Point(xy) for xy in zip(merged.lon, merged.lat)]
    gdf = gpd.GeoDataFrame(merged, geometry=geometry)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf.plot(column='Avg_PM25', ax=ax, legend=True, cmap='OrRd', markersize=25)
    plt.title("Average PM2.5 in 2022")
    plt.axis('off')
    plt.savefig(figures_dir / f"PM25_Map_2022.png")
    plt.close()

if __name__ == "__main__":
    base_path = Path("..")
    folder = base_path / "Raw" / "Air Pollution Data"
    village_csv = base_path / "Raw" / "village_locations.csv"

    # Toggle between Annual and Monthly here
    nc_dir = folder / "Monthly"  # for monthly
    #nc_dir = folder / "Annual"  # for annual

    tables_dir = base_path / "Tables"
    figures_dir = base_path / "Figures"

    process_pm25(
        village_csv_path=village_csv,
        nc_folder_path=nc_dir,
        output_dir=tables_dir,
        figures_dir=figures_dir,
        k_values=[3, 5, 10],
        years=list(range(1998, 2023)),
        is_monthly=True  # Change to False if using Annual folder
    )
