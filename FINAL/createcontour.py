import arcpy
from arcpy import env
from arcpy.sa import *

arcpy.CheckOutExtension("Spatial")
env.workspace = r"C:\Users\Shahzaib.Ahmed\Downloads"  # Use a local path, not managed by OneDrive
env.overwriteOutput = True

# Reading the .prj file to get the spatial reference
prj_file = r"C:\Users\Shahzaib.Ahmed\Downloads\diz.prj"
with open(prj_file, 'r') as file:
    wkt = file.read()
spatial_ref = arcpy.SpatialReference()
spatial_ref.loadFromString(wkt)

csv_file = r"C:\Users\Shahzaib.Ahmed\Downloads\output.csv"
x_coords = "X_M"
y_coords = "Y_M"
value_field = "NO2_MAXHOURLY_PPB"

# Create XY Layer from CSV
xy_layer = "in_memory/xy_layer"
arcpy.management.XYTableToPoint(csv_file, xy_layer, x_coords, y_coords, None, spatial_ref)

# Natural Neighbor Interpolation
interpolated_raster = NaturalNeighbor(xy_layer, value_field)

# Generating Contours directly from the in-memory raster
contour_interval = 10
contour_output = r"C:\Users\Shahzaib.Ahmed\Downloads\contour_output.shp"
arcpy.sa.Contour(interpolated_raster, contour_output, contour_interval)

# Cleanup
arcpy.management.Delete(xy_layer)
arcpy.CheckInExtension("Spatial")
