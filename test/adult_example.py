#!/usr/bin/python

import sys

## IMPORTS
import pandas as pd
import numpy as np
import cPickle
import os

from cleaningbenchmark.NoiseModels.RandomNoise import GaussianNoiseModel, ZipfNoiseModel, MissingNoiseModel
from cleaningbenchmark.NoiseModels.RandomNoise import CategoricalNoiseModel
from cleaningbenchmark.NoiseModels.RandomNoise import MixedNoiseTupleWiseModel
from cleaningbenchmark.NoiseModels.ConstraintModel import cCFDNoiseModel
from cleaningbenchmark.Utils.Utils import pd_df_diff
from collections import namedtuple


## Dataset Pre-Processing and Definitions


# User Definitions

# definitions for saving process
name_file = "adult"

# Data Path to Adult Datasets
DATADIR_PATH = "your data directory here, where we can find the adult dataset from UCI ML Repo"

# Load Clean Version of Adult Dataset from UCI ML Repository (.CSV file)
df_data = pd.read_csv(DATADIR_PATH + "/adult.csv")



# numerical features
num_feat_names = ['age', 
                  'fnlwgt', 
                  'hours-per-week', 
                  'education-num', 
                  'capital-gain', 
                  'capital-loss']

# categorical features
cat_feat_names = ['bracket-salary',
                  'native-country',
                  'sex',
                  'race',
                  'relationship',
                  'occupation',
                  'marital-status',
                  'education',
                  'workclass']


# Automated Definitions

# convert categorical features to categorical type
df_data[cat_feat_names] = df_data[cat_feat_names].apply(lambda x: x.astype('category'))



# ---- define helper structs for Categorical Noise Model (order of features matters) ----

# dictionary of lists, for each feature. List is the possible categories for that feature
category_dict = dict([(cat_name, [x for x in df_data[cat_name].cat.categories.tolist() if x != '?']) 
                     for cat_name in cat_feat_names])

cats_names = [category_dict[col] for col in cat_feat_names]

cats_probs = [np.array(df_data[col].value_counts()[category_dict[col]].values / float(df_data[col].value_counts()[category_dict[col]].sum()))
              for col in cat_feat_names]



# ---- define helper structs for Numerical Noise Model (order of features matters) ----
means_df = df_data[num_feat_names].mean().values # means of numerical features
stds_df = df_data[num_feat_names].std().values # standard deviations of numerical features



# ---- define helper structs for Mixed (Numerical + Categorical) Noise Model (order of features matters) ----

# dictionaries mapping the indexes between full dataframe and helper structures
idx_map_cat = dict([(df_data.columns.get_loc(col), i) for i, col in enumerate(cat_feat_names)])
idx_map_num = dict([(df_data.columns.get_loc(col), i) for i, col in enumerate(num_feat_names)])
# boolean array defining which features are categories (following the same order as the dataframe)
cat_array_bool = (df_data.dtypes.apply(lambda t: str(t)) == 'category').values



### DEFINE AND APPLY NOISE MODELS ###

# 1)
## Generate Adult 'Overt Outliers' Toy Example

# Define Categorical Noise Model
cat_mdl_overt = CategoricalNoiseModel((df_data.shape[0], len(cat_feat_names)), cats_names, 
                                cats_probs_list=[], typo_prob=0.70) 

# Define Numerical Noise Model
num_mdl_overt = ZipfNoiseModel((df_data.shape[0], len(num_feat_names)), z=1.8, scale=stds_df,
                         int_cast=np.ones(len(num_feat_names), dtype=bool), active_neg=True)

# Define Mixed Model (0.10*0.10 = 0.01 of cells noised, and 0.10 of tuples)
mix_mdl_overt = MixedNoiseTupleWiseModel(df_data.shape, cat_array_bool, 
                                         idx_map_cat, idx_map_num, cat_mdl_overt,
                                         num_mdl_overt, probability=0.10, p_row=0.10)

# Apply Noising of Data
noised_overt_outliers, _ = mix_mdl_overt.apply(df_data.values)
noised_overt_outliers_df = pd.DataFrame(noised_overt_outliers, columns=df_data.columns, index=df_data.index)

# Get Ground-Truth
df_changes_overt, cell_changed_overt, tuples_changed_overt = pd_df_diff(df_data, noised_overt_outliers_df)

print "Changes to Adult Dataset: Overt Outliers"
print df_changes_overt
print "\n\n\n\n"

# Save Data
folder_path = DATADIR_PATH + "/overt_noise/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
noised_overt_outliers_df.to_csv(folder_path + name_file + "_noised_overt.csv")

df_changes_overt.to_csv(folder_path + name_file + "_df_changes_noise_overt_gt_errors.csv")



# 2)
## Generate Adult 'Covert Outliers' Toy Example

# Define Categorical Noise Model
cat_mdl_covert = CategoricalNoiseModel((df_data.shape[0], len(cat_feat_names)), cats_names, 
                                       cats_probs_list=cats_probs, typo_prob=0.30) 

# Define Numerical Noise Model
num_mdl_covert = ZipfNoiseModel((df_data.shape[0], len(num_feat_names)), z=3, scale=stds_df/2,
                                int_cast=np.ones(len(num_feat_names), dtype=bool), active_neg=True)

# Define Mixed Model (0.10*0.10 = 0.01 of cells noised, and 0.10 of tuples)
mix_mdl_covert = MixedNoiseTupleWiseModel(df_data.shape, cat_array_bool, 
                                          idx_map_cat, idx_map_num, cat_mdl_covert,
                                          num_mdl_covert, probability=0.10, p_row=0.10)

# Apply Noising of Data
noised_covert_outliers, _ = mix_mdl_covert.apply(df_data.values)
noised_covert_outliers_df = pd.DataFrame(noised_covert_outliers, columns=df_data.columns, index=df_data.index)

# Get Ground-Truth
df_changes_covert, cell_changed_covert, tuples_changed_covert = pd_df_diff(df_data, noised_covert_outliers_df)

print "Changes to Adult Dataset: Covert Outliers"
print df_changes_covert
print "\n\n\n\n"

# Save Data
folder_path = DATADIR_PATH + "/covert_noise/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
noised_covert_outliers_df.to_csv(folder_path + name_file + "_noised_covert.csv")

df_changes_covert.to_csv(folder_path + name_file + "_df_changes_noise_covert_gt_errors.csv")

