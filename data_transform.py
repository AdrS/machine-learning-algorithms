import numpy as np

def pick_smallest_datatypes(data):
    '''Picks smallest types that can hold the data values.'''
    # Inspired by
    # https://www.kaggle.com/code/nickycan/compress-70-of-dataset/notebook
    limits = {
        np.float16: (np.finfo(np.float16).min, np.finfo(np.float16).max),
        np.float32: (np.finfo(np.float32).min, np.finfo(np.float32).max),
        np.int8: (np.iinfo(np.int8).min, np.iinfo(np.int8).max),
        np.int16: (np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        np.int32: (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
    }
    def in_range(minimum, maximum, dtype):
        dtype_min, dtype_max = limits[dtype]
        return dtype_min <= minimum and maximum <= dtype_max
    for col in data.columns:
        dtype = data[col].dtype
        if dtype == 'float64':
            minimum = data[col].min()
            maximum = data[col].max()
            if in_range(minimum, maximum, np.float16):
              data[col] = data[col].astype(np.float16)
            elif in_range(minimum, maximum, np.float32):
              data[col] = data[col].astype(np.float32)
        elif dtype == 'int64':
            minimum = data[col].min()
            maximum = data[col].max()
            if in_range(minimum, maximum, np.int8):
              data[col] = data[col].astype(np.int8)
            elif in_range(minimum, maximum, np.int16):
              data[col] = data[col].astype(np.int16)
            elif in_range(minimum, maximum, np.int32):
              data[col] = data[col].astype(np.int32)
