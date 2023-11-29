import laspy
# conda install geopandas=0.8.1
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import pickle
import os
import csv

def read_las(output_las_path):
    las = laspy.read(output_las_path)
    return las

def convert_las_to_df(las):
    # Convert las file to numpy array
    lidar_points = np.array((las.x, las.y, las.z, las.intensity, las.raw_classification, las.scan_angle_rank)).transpose()
    # Transform numpy array to pandas DataFrame
    lidar_df = pd.DataFrame(lidar_points)
    return lidar_df

def geo_reference_df(lidar_df):
    crs = None
    geometry_3D = [shapely.geometry.Point(xyz) for xyz in zip(lidar_df[0], lidar_df[1], lidar_df[2])]
    lidar_3D_gdf = gpd.GeoDataFrame(geometry_3D, crs=crs, geometry=geometry_3D)
    lidar_3D_gdf.crs = {'init': 'epsg:27700'}  # set correct spatial reference
    lidar_3D_gdf = lidar_3D_gdf.drop(columns=[0])
    lidar_3D_gdf = lidar_3D_gdf.to_crs(4326)
    assert lidar_3D_gdf.loc[0, 'geometry'].z != np.nan, "LiDAR points are missing z-coordinate"
    lidar_3D_gdf['2D_geom'] = [shapely.geometry.Point(xy) for xy in
                               zip(lidar_3D_gdf.geometry.x, lidar_3D_gdf.geometry.y)]
    return lidar_3D_gdf

def subsample_lidar_points(lidar_df):
    lidar_subset = lidar_df.sample(n=10000000, random_state=1)
    lidar_subset = lidar_subset.reset_index(drop=True)
    return lidar_subset

def convert_lidar_las_to_gdf(las_file_path, test_run=True):

    las_data = read_las(las_file_path)
    lidar_df = convert_las_to_df(las_data)
    if test_run:
        lidar_df = subsample_lidar_points(lidar_df)
    lidar_3D_gdf = geo_reference_df(lidar_df)
    return lidar_3D_gdf

def read_building_footprints(input_path):
    buidling_footprints = gpd.read_file(input_path)
    buidling_footprints = buidling_footprints.loc[buidling_footprints.geometry.apply(lambda x: isinstance(x, shapely.geometry.polygon.Polygon))]
    return buidling_footprints

def filter_points_per_footprint(building_footprint_path, lidar_3D_gdf):
    buidling_footprints = read_building_footprints(building_footprint_path)
    lidar_3D_gdf = lidar_3D_gdf.set_geometry('2D_geom')
    # Spatial join operation drops the 2D_geom column
    points_per_footprint = gpd.sjoin(buidling_footprints, lidar_3D_gdf, how="inner", op='intersects')
    points_per_footprint = points_per_footprint.set_geometry('geometry_right')
    return points_per_footprint

def dissolve_points_per_footprint(points_per_footprint):

    points_per_footprint['index_id'] = points_per_footprint.index
    points_per_footprint_dissolved = points_per_footprint.dissolve('index_id')
    points_per_footprint_dissolved = points_per_footprint_dissolved.rename_geometry('points_per_footprint')
    points_per_footprint_dissolved = points_per_footprint_dissolved.rename(
        columns={'geometry_left': 'footprint_polygon'})
    points_per_footprint_dissolved = points_per_footprint_dissolved.reset_index(drop=True)
    points_per_footprint_dissolved = points_per_footprint_dissolved.loc[
        points_per_footprint_dissolved.geometry.apply(lambda x: isinstance(x, shapely.geometry.MultiPoint))]
    points_per_footprint_dissolved['num_points'] = points_per_footprint_dissolved.apply(
        lambda row: len(row.points_per_footprint), axis=1)

    return points_per_footprint_dissolved

def convert_points_to_epsg_27700(points_per_footprint_dissolved):
    points_per_footprint_dissolved = points_per_footprint_dissolved.to_crs(27700)
    points_per_footprint_dissolved = points_per_footprint_dissolved.set_geometry('footprint_polygon')
    points_per_footprint_dissolved['footprint_polygon'] = points_per_footprint_dissolved['footprint_polygon'].set_crs(
        'epsg:4326', allow_override=True)
    points_per_footprint_dissolved = points_per_footprint_dissolved.to_crs(27700)
    return points_per_footprint_dissolved

def compute_footprint_boundaries(points_per_footprint_dissolved):
    # Add footprint bounds: minx, miny, maxx, maxy
    points_per_footprint_dissolved['footprint_bounds'] = points_per_footprint_dissolved['footprint_polygon'].apply(
        lambda footprint: footprint.bounds)
    # expand footprint polygon bounds into its own dataframe
    bounds = points_per_footprint_dissolved['footprint_bounds'].apply(pd.Series)
    bounds = bounds.rename(columns={0: 'footprint_minx', 1: 'footprint_miny', 2: 'footprint_maxx', 3: 'footprint_maxy'})
    points_per_footprint_with_boundaries = pd.concat([points_per_footprint_dissolved, bounds], axis=1)
    return points_per_footprint_with_boundaries

def max_z(row):
    max_z = 0
    for elem in list(row.points_per_footprint.geoms):
        if elem.z > max_z:
            max_z = elem.z
    return max_z

def min_z(row):
    min_z = 999
    for elem in list(row.points_per_footprint.geoms):
        if elem.z < min_z:
            min_z = elem.z
    return min_z

def point_to_numpy(row):
    point_list = []
    for point in list(row.points_per_footprint.geoms):
        point_list.append([point.x, point.y, point.z])
    return point_list

def normalize_points(row):
    normalized_points = []
    footprint_minx = row['footprint_minx']
    footprint_miny = row['footprint_miny']
    footprint_maxx = row['footprint_maxx']
    footprint_maxy = row['footprint_maxy']

    for point in list(row.unnormalized_points):
        normalized_x = (point[0] - footprint_minx) / (footprint_maxx - footprint_minx)
        normalized_y = (point[1] - footprint_miny) / (footprint_maxy - footprint_miny)
        normalized_z = (point[2] - row.min_z) / ((row.max_z - row.min_z) + 0.0000001)
        normalized_points.append([normalized_x, normalized_y, normalized_z])
    normalized_points_np = np.asarray(normalized_points, dtype=np.float32)
    return normalized_points_np

def min_max_scale_points_within_each_footprint(points_per_footprint_with_boundaries):

    # Min-Max-Scale each point within a respective building footprint to lie between 0 and 1
    points_per_footprint_with_boundaries['max_z'] = points_per_footprint_with_boundaries.apply(max_z, axis=1)
    points_per_footprint_with_boundaries['min_z'] = points_per_footprint_with_boundaries.apply(min_z, axis=1)
    points_per_footprint_with_boundaries['mean_z'] = points_per_footprint_with_boundaries.apply(lambda row: (row.max_z + row.min_z) / 2, axis=1)
    points_per_footprint_with_boundaries['unnormalized_points'] = points_per_footprint_with_boundaries.apply(point_to_numpy, axis=1)
    points_per_footprint_with_boundaries['normalized_points'] = points_per_footprint_with_boundaries.apply(normalize_points, axis=1)
    return points_per_footprint_with_boundaries

def sample_a_thousand_points(row):
    number_of_points_available = row['normalized_points'].shape[0]
    select_random_subsample = np.random.randint(number_of_points_available, size=1000)
    sub_sampled_point_cloud = row['normalized_points'][select_random_subsample, :]
    return sub_sampled_point_cloud

def sample_constant_number_of_points_per_building(scaled_points_per_footprint):

    pc_gdf = scaled_points_per_footprint.loc[scaled_points_per_footprint.num_points > 1000]
    if len(pc_gdf.index) > 0:
        pc_gdf['sub_sampled_point_cloud'] = pc_gdf.apply(sample_a_thousand_points, axis=1)
        pc_gdf = pc_gdf.to_crs(4326)
        pc_gdf['footprint_centroid'] = pc_gdf['footprint_polygon'].centroid
        pc_gdf = pc_gdf.reset_index(drop=True)
    return pc_gdf

def main():

    # Configuration
    # /Users/kevin/cs236g/Raw_LAS
    # /home/groups/fischer/raw_las_data
    las_input_dir = "/home/groups/fischer/raw_las_data"
    # /Users/kevin/cs236g/batch_processed_las_data
    # /home/users/kdmayer/CS236G_Course_Project/batch_processed_las_data
    processed_las_dir = "/home/users/kdmayer/CS236G_Course_Project/batch_processed_las_data"
    # /Users/kevin/cs236g/coventry_building_footprints.geojson
    # /home/users/kdmayer/CS236G_Course_Project/coventry_building_footprints.geojson
    building_footprint_path = "/home/users/kdmayer/CS236G_Course_Project/coventry_building_footprints.geojson"
    # /Users/kevin/cs236g/batch_processed_numpy_data
    # /home/users/kdmayer/CS236G_Course_Project/batch_processed_data
    npy_dir = "/home/users/kdmayer/CS236G_Course_Project/batch_processed_numpy_data"
    test_run = False

    for las_file in os.listdir(las_input_dir):

        if las_file.endswith(".las"):

            print(f"Currently processing: {las_file}")

            # Transform point cloud into GeoDataFrame format
            lidar_3D_gdf = convert_lidar_las_to_gdf(os.path.join(las_input_dir, las_file), test_run=test_run)

            # Conduct spatial join to create one point cloud per building footprint
            points_per_footprint = filter_points_per_footprint(building_footprint_path, lidar_3D_gdf)

            # Aggregate all points within one building footprint into a geo-referenced collection
            points_per_footprint_dissolved = dissolve_points_per_footprint(points_per_footprint)
            points_per_footprint_epsg_27700 = convert_points_to_epsg_27700(points_per_footprint_dissolved)

            # Determine the footprint polygon bounds for each building
            points_per_footprint_with_boundaries = compute_footprint_boundaries(points_per_footprint_epsg_27700)

            # Min-Max-Scale each point within a respective building footprint to lie between 0 and 1
            scaled_points_per_footprint = min_max_scale_points_within_each_footprint(points_per_footprint_with_boundaries)

            save_path = os.path.join(processed_las_dir, las_file)

            print(f"Script executed successfully. Save GDF to pickle at {save_path}")

            with open(save_path, 'wb') as fp:
                pickle.dump(scaled_points_per_footprint, fp)

            pc_gdf = sample_constant_number_of_points_per_building(scaled_points_per_footprint)

            # If dataframe is empty, go to the next lidar tile
            if len(pc_gdf.index) == 0:
                continue

            print(f"Save point cloud dataset at {npy_dir}")
            print("*****************")
            print(f"Add {len(pc_gdf.index)} building point clouds")
            print("*****************")

            for idx, row in pc_gdf.iterrows():

                lon, lat = row['footprint_centroid'].coords.xy
                file_name = str(lon[0]) + ',' + str(lat[0])
                path = os.path.join(npy_dir, file_name)
                np.save(path, row['sub_sampled_point_cloud'])

            with open(os.path.join(processed_las_dir, "processed_las_files.csv"), "a") as csvFile:
                writer = csv.writer(csvFile, lineterminator="\n")
                writer.writerow([las_file])

            #os.remove(os.path.join(las_input_dir, las_file))

if __name__ == "__main__":
   main()
