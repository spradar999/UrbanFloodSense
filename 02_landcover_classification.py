import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Configuration: Update these paths ---
RASTER_PATH = '/Users/prakash/Documents/Prakash/Bengaluru_Planet/April22_2025_psscene_analytic_8b_sr_udm2/composite.tif'
IMPERVIOUS_SHP_PATH = '/Users/prakash/Documents/Prakash/TF_UB/landuse_vectors_osm/cleaned/for training/impervious_utm.shp'
PERVIOUS_SHP_PATH = '/Users/prakash/Documents/Prakash/TF_UB/landuse_vectors_osm/cleaned/for training/pervious_utm.shp'
OUTPUT_RASTER_PATH = '/Users/prakash/Documents/Prakash/TF_UB/planet/classified_output.tif'
CHUNK_SIZE = 512

# --- 2. Optimized Data Preparation using Rasterization ---
def prepare_training_data_optimized():
    """
    Optimized function to extract training data.
    1. Rasterizes all polygons into a single label mask.
    2. Reads the image and mask chunk-by-chunk to extract pixels.
    This is much faster than masking with thousands of individual polygons.
    """
    print("Step 2 (Optimized): Preparing training data via rasterization...")
    
    with rasterio.open(RASTER_PATH) as src:
        # Read shapefiles and ensure CRS matches the raster
        impervious_gdf = gpd.read_file(IMPERVIOUS_SHP_PATH).to_crs(src.crs)
        pervious_gdf = gpd.read_file(PERVIOUS_SHP_PATH).to_crs(src.crs)

        # Create tuples of (geometry, label_value) for rasterization
        impervious_shapes = [(geom, 1) for geom in impervious_gdf.geometry]
        pervious_shapes = [(geom, 0) for geom in pervious_gdf.geometry]
        
        # Combine shapes
        all_shapes = impervious_shapes + pervious_shapes
        
        print(f"Rasterizing {len(all_shapes)} polygons into a label mask...")
        # Rasterize polygons into a memory array. `fill=-1` is the background value.
        label_mask = rasterize(
            shapes=all_shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=-1,  # Use -1 for pixels outside any polygon
            dtype='int16'
        )
        
        print("Extracting pixels using the rasterized label mask...")
        # Now, iterate through the raster and the mask in chunks
        X_list, y_list = [], []
        for i in range(0, src.width, CHUNK_SIZE):
            for j in range(0, src.height, CHUNK_SIZE):
                width = min(CHUNK_SIZE, src.width - i)
                height = min(CHUNK_SIZE, src.height - j)
                window = Window(i, j, width, height)
                
                # Read the label mask chunk
                labels_chunk = label_mask[j:j+height, i:i+width]
                
                # Check if there are any labeled pixels in this chunk
                if np.any(labels_chunk != -1):
                    # Read the corresponding raster chunk
                    raster_chunk = src.read(window=window)
                    
                    # Reshape and filter
                    labels_flat = labels_chunk.flatten()
                    pixels_flat = raster_chunk.reshape(src.count, -1).T
                    
                    # Create a mask for valid (labeled) pixels
                    valid_mask = labels_flat != -1
                    
                    if np.any(valid_mask):
                        X_list.append(pixels_flat[valid_mask])
                        y_list.append(labels_flat[valid_mask])

        if not X_list:
            print("No training data found. Check if polygons overlap with the raster.")
            return None, None

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        print(f"Extracted {X.shape[0]} valid pixels for training.")
        return X, y

# --- 3. Model Training (No changes needed) ---
def train_model(X, y):
    """
    Splits data and trains a Random Forest classifier.
    """
    if X is None or y is None:
        print("Cannot train model due to lack of data.")
        return None, None
        
    print("\nStep 3: Training Random Forest model...")
    # Split into 70% training and 30% temporary (for test/validation)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Split the 30% temporary set into 20% test and 10% validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
    )

    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    # --- 4. Accuracy Assessment ---
    print("\nStep 4: Performing accuracy assessment on validation data (10%)...")
    y_pred_val = rf_classifier.predict(X_val)
    
    report = classification_report(y_val, y_pred_val)
    matrix = confusion_matrix(y_val, y_pred_val)
    
    print("Classification Report:")
    print(report)
    
    print("Confusion Matrix:")
    print(matrix)

    # Save metrics to a text file
    with open("classification_results.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(matrix))
    
    print(f"\nAccuracy metrics saved to classification_results.txt")
    
    return rf_classifier

# --- 5. Prediction on Full Image (No changes needed) ---
def predict_full_image(classifier, raster_path, output_path):
    """
    Performs prediction on the entire raster in a memory-efficient,
    chunk-by-chunk manner.
    """
    if not classifier:
        print("Classifier is not trained. Aborting prediction.")
        return
        
    print("\nStep 5: Performing prediction on the full image...")
    
    with rasterio.open(raster_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='uint8', nodata=255)
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(0, src.width, CHUNK_SIZE):
                for j in range(0, src.height, CHUNK_SIZE):
                    width = min(CHUNK_SIZE, src.width - i)
                    height = min(CHUNK_SIZE, src.height - j)
                    window = Window(i, j, width, height)
                    
                    chunk = src.read(window=window)
                    
                    nodata_val = src.nodata
                    if nodata_val is not None:
                        mask = np.any(chunk == nodata_val, axis=0)
                    else:
                        mask = np.zeros((height, width), dtype=bool)

                    pixels = chunk.reshape(src.count, -1).T
                    
                    result = np.full(pixels.shape[0], meta['nodata'], dtype='uint8')
                    valid_pixels_mask = ~np.any(pixels == nodata_val, axis=1) if nodata_val else np.ones(pixels.shape[0], dtype=bool)
                    
                    if np.any(valid_pixels_mask):
                        result[valid_pixels_mask] = classifier.predict(pixels[valid_pixels_mask])
                    
                    classified_chunk = result.reshape(height, width)
                    dst.write(classified_chunk, window=window, indexes=1)

                    print(f"Processed prediction chunk at window: {window}", end='\r')

    print("\nPrediction complete. Output saved to:", output_path)

# --- Main Execution ---
if __name__ == '__main__':
    # Use the new optimized function
    X_features, y_labels = prepare_training_data_optimized()
    if X_features is not None and y_labels is not None:
        trained_classifier = train_model(X_features, y_labels)
        predict_full_image(trained_classifier, RASTER_PATH, OUTPUT_RASTER_PATH)

