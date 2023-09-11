import os
from osgeo import gdal


def read_GeoTiff(file_dir):
    """Read a GeoTiff file as numpy array

    Args:
        file_dir (str): image filepath

    Returns:
        _type_: _description_
    """

    img = gdal.Open(file_dir)

    img_Nrows = img.RasterYSize
    img_Ncols = img.RasterXSize
    img_Nbands = img.RasterCount
    img_GeoTransform = img.GetGeoTransform()
    img_Projection = img.GetProjection()

    img_arr = img.ReadAsArray(0, 0, img_Ncols, img_Nrows) # xoff, yoff, xcount, ycount
    img = None

    return img_arr, img_Nrows, img_Ncols, img_Nbands, img_GeoTransform, img_Projection


# Write a classification map to GeoTIFF file
def Write_GeoTiff(array, filename, Nrows, Ncols, Nbands, geotransform=None, projection=None):
    driver = gdal.GetDriverByName('GTiff')
    
    dataset_output = driver.Create(filename, Ncols, Nrows, Nbands, gdal.GDT_Int32)#gdal.GDT_Float32)
    for b in range(0, Nbands):
        dataset_output.GetRasterBand(b+1).WriteArray(array[b])
    
    if geotransform is not None:
        gt = list(geotransform)
        dataset_output.SetGeoTransform(tuple(gt))
    
    if projection is not None:        
        dataset_output.SetProjection(projection)
    
    dataset_output = None

