
import pandas as pd
import os

# returns a dict mapping subclass_index (int) -> superclass_index (int)
def load_hierarchy_mapping(train_csv_path):
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")
        
    df = pd.read_csv(train_csv_path)
    
    # unique pairs of (subclass_index, superclass_index)
    mapping_df = df[['subclass_index', 'superclass_index']].drop_duplicates()
    
    # converting to dictionary
    mapping_dict = dict(zip(mapping_df['subclass_index'], mapping_df['superclass_index']))
    
    return mapping_dict

# returns a function
# which returns True if (super_idx, sub_idx) is valid in the map

# NOTE: indices are >= 0 (Novel classes not handled)
def get_consistency_checker(mapping_dict):
    def check(super_idx, sub_idx):
        
        # should never happen
        if sub_idx not in mapping_dict:
            return False 
            
        expected_super = mapping_dict[sub_idx]
        return super_idx == expected_super

    return check