import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# 1. Input/Output Paths
# Path to high-resolution multi-spectral imagery (e.g., PlanetScope 8-band)
RASTER_PATH = 'data/planet_composite.tif' 

# Path to training labels (Vector/Shapefile)
IMPERVIOUS_SHP_PATH = 'vectors/impervious_training.shp'
PERVIOUS_SHP_PATH   = 'vectors/pervious_training.shp'

# Destination for the classified landcover map
OUTPUT_RASTER_PATH  = 'outputs/classified_landcover.tif'
METRICS_OUTPUT_PATH = 'outputs/classification_report.txt'

# 2. Model Hyperparameters (Random Forest)
RF_ESTIMATORS = 100       # Number of trees in the forest
RANDOM_STATE  = 42        # Ensures reproducible results
N_JOBS        = -1        # Use all available CPU cores for training

# 3. Data Processing Settings
CHUNK_SIZE  = 512         # Window size for memory-efficient processing
TRAIN_SPLIT = 0.7         # 70% for training
TEST_SPLIT  = 0.2         # 20% for final testing
VAL_SPLIT   = 0.1         # 10% for validation during training

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def prepare_training_data_optimized():
    """
    Extracts spectral signatures from the raster based on provided polygons.
    Uses rasterization for high-performance pixel extraction.
    """
    print(f"Loading data and rasterizing polygons from {IMPERVIOUS_SHP_PATH}...")
    
    with rasterio.open(RASTER_PATH) as src:
        # Load and align CRS
        imp_gdf = gpd.read_file(IMPERVIOUS_SHP_PATH).to_crs(src.crs)
        per_gdf = gpd.read_file(PERVIOUS_SHP_PATH).to_crs(src.crs)

        # Labeling: Impervious = 1, Pervious = 0
        all_shapes = [(geom, 1) for geom in imp_gdf.geometry] + \
                     [(geom, 0) for geom in per_gdf.geometry]
        
        label_mask = rasterize(
            shapes=all_shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=-1, # Background
            dtype='int16'
        )
        
        X_list, y_list = [], []
        for i in range(0, src.width, CHUNK_SIZE):
            for j in range(0, src.height, CHUNK_SIZE):
                w = min(CHUNK_SIZE, src.width - i)
                h = min(CHUNK_SIZE, src.height - j)
                window = Window(i, j, w, h)
                
                labels_chunk = label_mask[j:j+h, i:i+w]
                if np.any(labels_chunk != -1):
                    raster_chunk = src.read(window=window)
                    pixels = raster_chunk.reshape(src.count, -1).T
                    labels = labels_chunk.flatten()
                    
                    valid = labels != -1
                    X_list.append(pixels[valid])
                    y_list.append(labels[valid])

        if not X_list:
            raise ValueError("No overlapping training data found between raster and vectors.")

        return np.vstack(X_list), np.concatenate(y_list)

def train_and_validate(X, y):
    """
    Trains the Random Forest model and performs a 3-way split evaluation.
    """
    print("\nTraining Random Forest model...")
    # Stratified splits to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1-TRAIN_SPLIT), random_state=RANDOM_STATE, stratify=y)
    
    # Calculate relative test/val split from the remainder
    test_ratio = TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(1-test_ratio), random_state=RANDOM_STATE, stratify=y_temp)

    clf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    clf.fit(X_train, y_train)
    
    # Accuracy Assessment
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred)
    
    print("Validation Results:\n", report)
    with open(METRICS_OUTPUT_PATH, "w") as f:
        f.write(f"Classification Report (Val Split: {VAL_SPLIT})\n" + report)
    
    return clf

def apply_classification(clf, input_path, output_path):
    """
    Applies the trained classifier to the full scene using windowed reading.
    """
    print(f"\nClassifying full image: {input_path}")
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', nodata=255)
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(0, src.width, CHUNK_SIZE):
                for j in range(0, src.height, CHUNK_SIZE):
                    w, h = min(CHUNK_SIZE, src.width-i), min(CHUNK_SIZE, src.height-j)
                    window = Window(i, j, w, h)
                    
                    chunk = src.read(window=window)
                    pixels = chunk.reshape(src.count, -1).T
                    
                    # Handle NoData if present
                    res = np.full(pixels.shape[0], meta['nodata'], dtype='uint8')
                    valid = ~np.any(pixels == src.nodata, axis=1) if src.nodata else np.ones(pixels.shape[0], dtype=bool)
                    
                    if np.any(valid):
                        res[valid] = clf.predict(pixels[valid])
                    
                    dst.write(res.reshape(h, w), window=window, indexes=1)
                    print(f"Processing... {i/src.width*100:.1f}% complete", end='\r')

if __name__ == '__main__':
    try:
        features, labels = prepare_training_data_optimized()
        model = train_and_validate(features, labels)
        apply_classification(model, RASTER_PATH, OUTPUT_RASTER_PATH)
        print("\nPipeline execution successful.")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
