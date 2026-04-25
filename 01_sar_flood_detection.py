import ee
ee.Initialize()

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
# 1. Spatial & Temporal Constraints
# Replace ROI_TABLE_PATH with your GEE Asset path if not using the 'table' variable
ROI_TABLE_PATH = 'projects/your-project/assets/bengaluru_geometry' 
BEFORE_START = '2021-11-05'    # Start date for "dry" reference baseline
BEFORE_END   = '2021-11-20'    # End date for "dry" reference baseline
AFTER_START  = '2021-11-21'    # Start date for "flood" event analysis
AFTER_END    = '2021-11-25'    # End date for "flood" event analysis

# 2. Algorithm Thresholds
DIFF_THRESHOLD = 1.5           # Ratio threshold for change detection (VH polarization)
SLOPE_THRESHOLD = 5            # Slope in degrees; pixels above this are masked
MIN_CONNECTED_PIXELS = 5       # Minimum cluster size (pixels) to remove noise

# 3. Export Settings
EXPORT_FOLDER = 'Bengaluru_Floods'           # Google Drive destination folder
EXPORT_NAME   = 'Flooded_Area_Bengaluru_SAR' # File prefix
EXPORT_SCALE  = 10                           # Spatial resolution (meters)
EXPORT_CRS    = 'EPSG:4326'                  # Target coordinate reference system

# -----------------------------
# Data Loading
# -----------------------------
# Note: 'table' is assumed to be defined in your GEE environment. 
# If running locally, define your geometry via ee.Geometry or ee.FeatureCollection.
geometry = table.geometry() 

# External Global Datasets
hydrosheds = ee.Image('WWF/HydroSHEDS/03VFDEM')
gsw = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')

# -----------------------------
# Sentinel-1 Collection Setup
# -----------------------------
collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filterBounds(geometry)
    .select('VH'))

# Create baseline (Before) and event (After) mosaics
before = collection.filterDate(BEFORE_START, BEFORE_END).mosaic().clip(geometry)
after = collection.filterDate(AFTER_START, AFTER_END).mosaic().clip(geometry)

# -----------------------------
# Image Processing Utilities
# -----------------------------
def toNatural(img):
    """Convert dB back to natural linear power."""
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    """Convert natural linear power to dB."""
    return ee.Image(img).log10().multiply(10.0)

def RefinedLee(img):
    """
    Applies the Refined Lee speckle filter to SAR data.
    Maintains edge integrity by using directional windowing.
    """
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0]
    ])
    sample_kernel = ee.Kernel.fixed(7,7,sample_weights,3,3,False)
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    max_gradient = gradients.reduce(ee.Reducer.max())
    gradmask = gradients.eq(max_gradient)
    gradmask = gradmask.addBands(gradmask)

    directions = sample_mean.select(1).subtract(sample_mean.select(4)) \
        .gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)

    directions = directions.addBands(
        sample_mean.select(6).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))

    directions = directions.addBands(
        sample_mean.select(3).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))

    directions = directions.addBands(
        sample_mean.select(0).subtract(sample_mean.select(4))
        .gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))

    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    directions = directions.updateMask(gradmask)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
    sigmaV = (sample_stats.toArray().arraySort().arraySlice(0, 0, 5).arrayReduce(ee.Reducer.mean(), [0]))

    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))
    diag_weights = ee.List([
        [1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]
    ])

    rect_kernel = ee.Kernel.fixed(7,7,rect_weights,3,3,False)
    diag_kernel = ee.Kernel.fixed(7,7,diag_weights,3,3,False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    for i in range(1,4):
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
    b = varX.divide(dir_var)
    result = dir_mean.add(b.multiply(img.subtract(dir_mean)))

    return result

# -----------------------------
# Flood Analysis Pipeline
# -----------------------------
# 1. Apply Despeckling
before_filtered = toDB(RefinedLee(toNatural(before)))
after_filtered = toDB(RefinedLee(toNatural(after)))

# 2. Difference Detection (Ratio method)
difference = after_filtered.divide(before_filtered)
flooded = difference.gt(DIFF_THRESHOLD).rename('water').selfMask()

# 3. Spatial Refinement & Masking
# Mask 1: Remove permanent water bodies
permanent_water = gsw.select('seasonality').gte(5).clip(geometry)
flooded = flooded.updateMask(permanent_water.unmask(0).Not())

# Mask 2: Remove steep terrain (slope > threshold)
terrain = ee.Algorithms.Terrain(hydrosheds)
slope = terrain.select('slope')
flooded = flooded.updateMask(slope.lte(SLOPE_THRESHOLD))

# Mask 3: Connected Component Filter (removes isolated noise)
connections = flooded.connectedPixelCount(25)
flooded = flooded.updateMask(connections.gte(MIN_CONNECTED_PIXELS))

# -----------------------------
# Analysis & Reporting
# -----------------------------
total_area = geometry.area().divide(10000) 

stats = flooded.multiply(ee.Image.pixelArea()).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geometry,
    scale=EXPORT_SCALE,
    maxPixels=1e10,
    tileScale=16
)

flood_area = ee.Number(stats.get('water')).divide(10000)

print(f'Total ROI Area (Ha): {total_area.getInfo():.2f}')
print(f'Flooded Area detected (Ha): {flood_area.getInfo():.2f}')

# -----------------------------
# Asset/Drive Export
# -----------------------------
task = ee.batch.Export.image.toDrive(
    image=flooded,
    description=EXPORT_NAME,
    folder=EXPORT_FOLDER,
    fileNamePrefix=EXPORT_NAME,
    region=geometry,
    scale=EXPORT_SCALE,
    crs=EXPORT_CRS,
    maxPixels=1e13
)

print(f'Task launched: Exporting to Drive folder "{EXPORT_FOLDER}"...')
task.start()
