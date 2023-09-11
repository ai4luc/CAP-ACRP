
import os
import pandas as pd
import numpy as np


def write_samples_to_file(df, dir_output, filename, target_col=None, target=None):

    #Include the class column in the dataframes
    if target_col is not None and target is not None:
        df[target_col] = target

    filepath = os.path.join(dir_output, filename)

    if not os.path.exists(dir_output):
        # NOTE makedirs is able to create folders and subfolders
        #os.mkdir(dir_output)
        os.makedirs(dir_output)
    
    # Writing sample data sets to files
    try:
        pd.DataFrame(df).to_csv(filepath, index=False, header=True)
    except Exception as e:
        print(str(e))


# Extract samples from a mask of samples
def extract_samples_from_mask(img_arr, mask_arr, classes, bands, class_col='target'):
    
    n_samples_class = []

    #df_indices = pd.DataFrame(np.hstack((np.indices(mask_arr.shape).reshape(2, mask_arr.size).T,\
    #                    mask_arr.reshape(-1, 1))), columns=['row', 'col', 'value'])
    #df_indices.reset_index(inplace=True)
    
    Ncols = mask_arr.shape[1]

    df_samples_all = pd.DataFrame(np.nan, index=range(0), columns=bands+[class_col])
    df_samples_all = df_samples_all.rename_axis(index='index', axis=0)
    for c, class_ in classes.items():
        indexes = np.where(mask_arr==c)
        
        if (indexes[0].size > 0):
            
            n_samples_class.append(indexes[0].size)
            df_temp = pd.DataFrame(np.nan, index=range(indexes[0].size), columns=bands+[class_col])
            
            df_temp[class_col] = c
            for b, band in enumerate(bands):
                a = img_arr[b, indexes[0], indexes[1]]
                df_temp[band] = pd.DataFrame(a)
            
            df_temp['index'] = pd.DataFrame((Ncols*indexes[0])+indexes[1])
            df_temp = df_temp.set_index(['index'])
            
            df_samples_all = pd.concat([df_samples_all, df_temp])
            
        else:
            n_samples_class.append(0)

    return df_samples_all, n_samples_class



