import ee
ee.Initialize()

# -----------------------------
# Load datasets
# -----------------------------
admin2 = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level2')
hydrosheds = ee.Image('WWF/HydroSHEDS/03VFDEM')
gsw = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')

# Dates
before_start = '2021-11-05'
before_end = '2021-11-20'
after_start = '2021-11-21'
after_end = '2021-11-25'

# Geometry (replace with your asset)
geometry = table.geometry()

# -----------------------------
# Sentinel-1 Collection
# -----------------------------
collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filterBounds(geometry)
    .select('VH'))

before_collection = collection.filterDate(before_start, before_end)
after_collection = collection.filterDate(after_start, after_end)

before = before_collection.mosaic().clip(geometry)
after = after_collection.mosaic().clip(geometry)

# -----------------------------
# Conversion Functions
# -----------------------------
def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    return ee.Image(img).log10().multiply(10.0)

# -----------------------------
# Refined Lee Filter
# -----------------------------
def RefinedLee(img):
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    sample_weights = ee.List([
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0],
        [0,1,0,1,0,1,0],
        [0,0,0,0,0,0,0]
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

    sigmaV = (sample_stats.toArray()
        .arraySort()
        .arraySlice(0, 0, 5)
        .arrayReduce(ee.Reducer.mean(), [0]))

    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(
                   ee.List.repeat(ee.List.repeat(1,7),4))

    diag_weights = ee.List([
        [1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0],
        [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0],
        [1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1]
    ])

    rect_kernel = ee.Kernel.fixed(7,7,rect_weights,3,3,False)
    diag_kernel = ee.Kernel.fixed(7,7,diag_weights,3,3,False)

    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(
        img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(
        img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    for i in range(1,4):
        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2*i+1)))

        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i))
            .updateMask(directions.eq(2*i+1)))

        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2*i+2)))

        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i))
            .updateMask(directions.eq(2*i+2)))

    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)) \
                  .divide(sigmaV.add(1.0))

    b = varX.divide(dir_var)
    result = dir_mean.add(b.multiply(img.subtract(dir_mean)))

    return result

# -----------------------------
# Apply Filter
# -----------------------------
before_filtered = toDB(RefinedLee(toNatural(before)))
after_filtered = toDB(RefinedLee(toNatural(after)))

difference = after_filtered.divide(before_filtered)

# Flood Detection
diff_threshold = 1.5
flooded = difference.gt(diff_threshold).rename('water').selfMask()

# Permanent Water Mask
permanent_water = gsw.select('seasonality').gte(5).clip(geometry)
flooded = flooded.updateMask(permanent_water.unmask(0).Not())

# Slope Mask
terrain = ee.Algorithms.Terrain(hydrosheds)
slope = terrain.select('slope')
flooded = flooded.updateMask(slope.lte(5))

# Remove Noise
connections = flooded.connectedPixelCount(25)
flooded = flooded.updateMask(connections.gte(5))

# -----------------------------
# Area Calculation
# -----------------------------
total_area = geometry.area().divide(10000)

stats = flooded.multiply(ee.Image.pixelArea()).reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geometry,
    scale=10,
    maxPixels=1e10,
    tileScale=16
)

flood_area = ee.Number(stats.get('water')).divide(10000)

print('Total Area (Ha):', total_area.getInfo())
print('Flooded Area (Ha):', flood_area.getInfo())

# -----------------------------
# Export
# -----------------------------
task = ee.batch.Export.image.toDrive(
    image=flooded,
    description='Flooded_Area_Bengaluru',
    folder='Bengaluru_Floods',
    fileNamePrefix='FA_Bengaluru_afterFiltered',
    region=geometry,
    scale=10,
    crs='EPSG:4326',
    maxPixels=1e13
)

task.start()