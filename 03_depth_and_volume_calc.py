# -*- coding: utf-8 -*-
"""
URBAN FLOOD DEPTH AND VOLUME QUANTIFICATION PIPELINE
Integrates Flood SAR masks, Landcover classifications, and Elevation Models.
"""

import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes, rasterize
from shapely.geometry import shape
from rasterstats import zonal_stats
from rasterio.warp import reproject, Resampling, calculate_default_transform

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. Primary Inputs
DEM_PATH        = "data/Dem.tif"        # High-res Digital Elevation Model
FLOOD_PIXEL_RAW = "data/Flood_Pixel.tif"     # Binary flood mask from SAR (Step 01)
IMPERVIOUS_MAP  = "data/Impervious.tif"      # Binary LULC map (Step 02)

# 2. Output CRS & Projection
# Use a projected CRS (e.g., UTM Zone 43N for Bengaluru) for accurate volume calculations
TARGET_CRS = "EPSG:32643" 

# 3. Intermediate & Final Outputs
VECTOR_FLOOD_PATH = "outputs/Flood_vectorized.shp"
ZONAL_MAX_PATH    = "outputs/zonal_max.shp"
DEM_MAX_RASTER    = "outputs/dem_max_surface.tif"
DEPTH_RASTER      = "outputs/flood_depth.tif"

# ==============================================================================
# STAGE 1: SPATIAL VECTORIZATION & ZONAL ANALYSIS
# ==============================================================================

def vectorize_and_zonal_max():
    """
    Converts flood pixels to polygons and extracts maximum elevation per patch
    to estimate the water surface level.
    """
    print("\nSTAGE 1: Vectorizing flood mask and calculating Zonal Max...")
    
    with rasterio.open(FLOOD_PIXEL_RAW) as src:
        band = src.read(1)
        mask = band > 0 # Assumes 1=Flooded, 0=Non-flooded
        
        results = (
            {'properties': {'maximum': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(band, mask=mask, transform=src.transform))
        )
        
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)
        gdf.to_file(VECTOR_FLOOD_PATH)

    # Calculate max elevation in each flood polygon (Water Surface Elevation)
    stats = zonal_stats(VECTOR_FLOOD_PATH, DEM_PATH, stats=["max"], nodata=None)
    gdf["max_elev"] = [s["max"] for s in stats]
    gdf.to_file(ZONAL_MAX_PATH)
    return gdf

# ==============================================================================
# STAGE 2: SURFACE RECONSTRUCTION & DEPTH CALCULATION
# ==============================================================================

def calculate_flood_depth(gdf):
    """
    Rasterizes the water surface elevation and subtracts the ground DEM.
    """
    print("STAGE 2: Calculating flood depth (Water Surface - Ground DEM)...")
    
    with rasterio.open(DEM_PATH) as src:
        meta = src.meta.copy()
        
        # Create Water Surface Elevation Raster (Rasterized Zonal Max)
        shapes_gen = [(geom, val) for geom, val in zip(gdf.geometry, gdf.max_elev) if val is not None]
        water_surface = rasterize(shapes_gen, out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype=meta["dtype"])
        
        # Depth = Surface - Terrain (Only where flooded)
        ground_dem = src.read(1)
        depth = np.where(water_surface > 0, water_surface - ground_dem, 0)
        depth = np.maximum(depth, 0) # Ensure no negative depth

        meta.update(dtype="float32", count=1)
        with rasterio.open(DEPTH_RASTER, "w", **meta) as dst:
            dst.write(depth.astype("float32"), 1)
            
    return DEPTH_RASTER

# ==============================================================================
# STAGE 3: VOLUMETRIC QUANTIFICATION BY LANDCOVER
# ==============================================================================

def quantify_volumes(depth_path):
    """
    Projects all layers to UTM and calculates total volume for Impervious/Pervious areas.
    """
    print("STAGE 3: Quantifying volumes in projected CRS...")
    
    with rasterio.open(depth_path) as src:
        # Calculate transform for Target Projection
        dst_trans, dst_w, dst_h = calculate_default_transform(src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        
        def reproject_layer(path, resampling=Resampling.bilinear):
            with rasterio.open(path) as s:
                out = np.zeros((dst_h, dst_w), dtype='float32')
                reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs, 
                          dst_transform=dst_trans, dst_crs=TARGET_CRS, resampling=resampling)
                return out

        # Reproject all required layers to the same grid
        depth_proj = reproject_layer(depth_path)
        imper_proj = reproject_layer(IMPERVIOUS_MAP, Resampling.nearest)
        
        # Calculate pixel area in meters squared
        pixel_area = abs(dst_trans[0] * dst_trans[4])
        
        # Masks
        is_flooded = depth_proj > 0
        is_imperv  = imper_proj == 1
        
        vol_imp = np.sum(depth_proj[is_flooded & is_imperv]) * pixel_area
        vol_per = np.sum(depth_proj[is_flooded & ~is_imperv]) * pixel_area
        
        print("\n" + "="*40)
        print(f"Projected CRS: {TARGET_CRS}")
        print(f"Pixel Resolution: {abs(dst_trans[0]):.2f}m x {abs(dst_trans[4]):.2f}m")
        print("-" * 40)
        print(f"Impervious Flood Volume: {vol_imp:,.2f} m³")
        print(f"Pervious Flood Volume:   {vol_per:,.2f} m³")
        print(f"Total Flood Volume:      {(vol_imp + vol_per):,.2f} m³")
        print("="*40)

if __name__ == '__main__':
    try:
        flood_gdf = vectorize_and_zonal_max()
        depth_file = calculate_flood_depth(flood_gdf)
        quantify_volumes(depth_file)
    except Exception as e:
        print(f"Analysis failed: {e}")
