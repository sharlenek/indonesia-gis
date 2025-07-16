import os
import rasterio
from rasterio.merge import merge
from shapely.geometry import box
from rtree import index
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging

# ---------------------- Configuration ----------------------
palm_tile_folder = "../../Raw/Palm Coverage/GlobalOilPalm_OP-YoP"
water_tile_folder = "../../Raw/Water Mask"
output_folder = "../../Processed/Aligned_Water_Mask"
os.makedirs(output_folder, exist_ok=True)

# ---------------------- Logging Setup ----------------------
log_file = os.path.join(output_folder, "alignment_log.txt")
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Starting alignment of JRC water masks to palm raster tiles...")

# ---------------------- Build Spatial Index for Water Tiles ----------------------
water_tile_paths = [os.path.join(water_tile_folder, f)
                    for f in os.listdir(water_tile_folder) if f.endswith('.tif')]
water_idx = index.Index()
water_bounds = {}

print(f"Found {len(water_tile_paths)} water tiles.")

for i, path in enumerate(water_tile_paths):
    try:
        with rasterio.open(path) as src:
            bounds = box(*src.bounds)
            water_bounds[i] = (path, bounds, src.crs)
            water_idx.insert(i, bounds.bounds)
    except Exception as e:
        logging.warning(f"Failed to index water tile {path}: {e}")

# ---------------------- Process Each Palm Tile ----------------------
palm_tile_paths = [os.path.join(palm_tile_folder, f)
                   for f in os.listdir(palm_tile_folder) if f.endswith('.tif')]
print(f"Found {len(palm_tile_paths)} palm tiles.")

for palm_path in palm_tile_paths:
    try:
        with rasterio.open(palm_path) as palm_src:
            palm_bounds = box(*palm_src.bounds)
            palm_crs = palm_src.crs
            palm_res = palm_src.res
            matched_indices = list(water_idx.intersection(palm_bounds.bounds))

            srcs_to_merge = []
            temp_memfiles = []

            palm_tile_name = os.path.basename(palm_path)
            logging.info(f"Processing palm tile: {palm_tile_name}")
            logging.info(f"Matched {len(matched_indices)} water tiles")
            print(f"→ {palm_tile_name}: matched {len(matched_indices)} water tiles")

            for i in matched_indices:
                water_path, bounds, water_crs = water_bounds[i]

                if water_crs != palm_crs:
                    logging.info(f"Reprojecting {os.path.basename(water_path)} from {water_crs} to {palm_crs}")
                    with rasterio.open(water_path) as water_src:
                        dst_transform, width, height = calculate_default_transform(
                            water_src.crs, palm_crs, water_src.width, water_src.height, *water_src.bounds)
                        kwargs = water_src.meta.copy()
                        kwargs.update({
                            'crs': palm_crs,
                            'transform': dst_transform,
                            'width': width,
                            'height': height
                        })
                        memfile = rasterio.io.MemoryFile()
                        temp_memfiles.append(memfile)
                        dst = memfile.open(**kwargs)
                        for j in range(1, water_src.count + 1):
                            reproject(
                                source=rasterio.band(water_src, j),
                                destination=rasterio.band(dst, j),
                                src_transform=water_src.transform,
                                src_crs=water_src.crs,
                                dst_transform=dst_transform,
                                dst_crs=palm_crs,
                                resampling=Resampling.nearest
                            )
                        dst.close()
                        srcs_to_merge.append(memfile.open())
                else:
                    srcs_to_merge.append(rasterio.open(water_path))

            # Merge and write result
            if srcs_to_merge:
                mosaic, out_trans = merge(srcs_to_merge, bounds=palm_src.bounds, res=palm_res)
                out_meta = palm_src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "count": 1
                })
                out_name = palm_tile_name.replace('.tif', '_water_mask.tif')
                out_path = os.path.join(output_folder, out_name)
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                logging.info(f"Saved aligned water mask to: {out_path}")
                print(f"✔️ Saved: {out_name}")
            else:
                logging.warning(f"No overlapping water tiles found for: {palm_tile_name}")
                print(f"⚠️ No water tiles for: {palm_tile_name}")

            # Clean up
            for src in srcs_to_merge:
                src.close()
            for memfile in temp_memfiles:
                memfile.close()

    except Exception as e:
        logging.error(f"Failed to process {palm_path}: {e}")
        print(f"❌ Error processing {palm_tile_name}: {e}")

logging.info("Finished aligning all JRC water masks.")
print("✅ Script finished.")
