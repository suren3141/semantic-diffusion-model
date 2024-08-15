import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from skimage.measure import regionprops
from tqdm import tqdm

def binary_to_instance(mask_path):
    try:
        # Load the mask image
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask_array = np.array(mask)
    except Exception as e:
        print(f"Error processing SHAPES for file {mask_path}: {e}")
        return None, None  # Or handle it appropriately
    
    # Binarize the mask (ensure binary values: 0 and 1)
    binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
    
    # Label connected components
    labeled_mask = measure.label(binary_mask)

def calculate_shape_properties(inst_mask):
    
    # Get properties of labeled regions
    properties = regionprops(inst_mask)
    
    if not properties:
        print("No shapes found in the mask.")
        mean_properties = {
            'area_weighted_solidity': 1,
            'total_area': 0,
            # 'area': [prop.area for prop in properties],
            'med_area': 0,
        }

        return mean_properties

        # return None
    
    # Initialize sums for each property
    sum_solidity = 0
    sum_area = 0
        
    # Accumulate values for each property
    for prop in properties:
        sum_solidity += prop.solidity * prop.area
        sum_area += prop.area
    
    # Calculate mean values for each property
    mean_properties = {
        'area_weighted_solidity': sum_solidity / sum_area,
        'total_area': sum_area,
        # 'area': [prop.area for prop in properties],
        'med_area': np.median([prop.area for prop in properties]),
    }
    
    return mean_properties


# from csbdeep.utils import Path, normalize
# from stardist import fill_label_holes
from tqdm.notebook import tqdm



def get_shape_properties_df(masks):

    y_prop = [calculate_shape_properties(y) for y in tqdm(masks)]

    keys = y_prop[0].keys()
    print(keys)

    df_prop = {k:[x[k] for x in y_prop] for k in keys}
    df_prop = {k:np.array(v) for k, v in df_prop.items()}
    df_prop = pd.DataFrame.from_dict(df_prop)

    return df_prop

from sklearn.preprocessing import MinMaxScaler

def get_prop_diff(df_sdm_prop, df_app_prop, quant=(.15, .85), weight = dict(med_area=.5, total_area=.5)):

    df_diff = {
        k : df_sdm_prop[k] - df_app_prop[k] for k in df_sdm_prop.keys()
    }
    df_diff = pd.DataFrame.from_dict(df_diff)

    df_diff_scaled = df_diff.abs()
    df_diff_scaled = df_diff_scaled.clip(upper=df_diff_scaled.quantile(quant[1]), axis=1)

    # print(df_diff_scaled.min(), df_diff_scaled.max())

    scaler = MinMaxScaler()
    df_diff_scaled = pd.DataFrame(scaler.fit_transform(df_diff_scaled), columns=df_diff_scaled.keys())

    # print(df_diff_scaled.min(), df_diff_scaled.max())

    df_diff_scaled['score'] = df_diff_scaled['med_area'] * .5 + df_diff_scaled['total_area'] * .5
    df_diff_scaled['prob'] = 1- df_diff_scaled['score'].clip(0, 1)

    return df_diff_scaled
