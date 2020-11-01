import ast
import os
import json
import rasterio
import pyproj
import geojson
import multiprocessing

import geopandas as gpd
import pandas as pd
import numpy as np
import numpy.ma as ma
import dask.dataframe as dd

from dask.multiprocessing import get
from functools import partial
from itertools import filterfalse
from glob import glob
from shapely.geometry import Point, Polygon, box, shape, mapping
from shapely.ops import transform
from shapely.wkt import dumps, loads
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling

import time

class MeasureDuration:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print("Total time taken: %s" % self.duration())

    def duration(self):
        return(str((self.end - self.start)) + ' seconds')

"""
Class for managing the Urban Atlas dataset.
The class is initialised with the directories containing the SAR and optical imagery in an already unzipped folder
"""
class UrbanAtlas:

    def __init__(self, opt_dir, sar_dir, cities=None, crs="EPSG:3035", load_geometry=False, workers=1, scheme="SURF", force=False, stats=False, resolution=2.5):
        self.opt_dir = opt_dir
        self.sar_dir = sar_dir
        self.scheme = scheme
        self.base_dir = os.path.commonpath([opt_dir, sar_dir])
        self.workers = workers if workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.resolution = resolution

        self.crs = pyproj.Proj(crs)

        self.city_list = self._create_city_list(cities)

        assert len(self.city_list) > 0, "No cities could be found, unable to continue"

        self.dataset = self._load_dataset()

        if stats:
            self.dataset.join(self.get_raster_stats())

        self.geometry = None
        if load_geometry:
            # print("WARNING WE ARE NOT LOADING EXISTING DATA")
            self.geometry = self.get_geometry(cities=self.city_list, reduce=True, reduction_set=["SAR", "OPT"], load_existing=not force)

    def _create_city_list(self, city_filter=None):
        # Build city sets from directory names
        opt_cities = set(map(os.path.basename, filter(os.path.isdir, glob(os.path.join(self.opt_dir, "*")))))
        sar_cities = set(map(os.path.basename, filter(os.path.isdir, glob(os.path.join(self.sar_dir, "*")))))

        cities = opt_cities.intersection(sar_cities)

        if city_filter:
            cities = cities.intersection(set(city_filter))

        return cities

    # Create a bounding box in global coordinates so we can store everything in a single geopandas dataframe
    def _create_bounding_box(self, raster):
        left, top = raster.bounds.left, raster.bounds.top
        right, bottom = raster.bounds.right, raster.bounds.bottom
        return box(left, top, right, bottom)

    def _calc_stats(self, row):
        mu = []
        std = []
        d_min = []
        d_max = []
        raster = row["raster"]
        for _, w in raster.block_windows():
            data = raster.read(window=w, masked=True)
            # SAR Images sometimes containing -1 which must have been an old nodataval
            data = np.ma.masked_less(data, 0)
            if not data.mask.all():
                mu.append(data.mean())
                std.append(data.std())
                d_min.append(data.min())
                d_max.append(data.max())
        return {"mu": np.mean(mu), "std": np.std(std), "min": np.min(d_min), "max": np.max(d_max)}

    def get_raster_stats(self):
        def apply_to_df(df):
            return df.apply(self._calc_stats, axis=1)

        ddata = dd.from_pandas(self.dataset, npartitions=self.workers)
        stats = ddata.map_partitions(apply_to_df, meta=dict).compute(scheduler="threads")
        stats = stats.apply(lambda s: pd.Series(s))
        return stats

    # EPSG:3035 is the projection used by all PRISM images but some ar fucked and need to be reprojected
    def _create_virtual_warped_raster(self, raster, resolution=2.5):

        # Broken PRISM images
        if raster.crs is None:
            raster._crs = self.crs

        if raster.crs is not None and raster.crs.to_epsg() == self.crs.crs.to_epsg() and raster.res == (resolution, resolution):
            return raster

        dst_transform, dst_width, dst_height = calculate_default_transform(raster.crs, "EPSG:{}".format(self.crs.crs.to_epsg()), raster.width, raster.height, *raster.bounds, resolution=(resolution, resolution))
        vrt_opts = {
            'resampling': Resampling.nearest,
            'crs': "EPSG:{}".format(self.crs.crs.to_epsg()),
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            }

        return WarpedVRT(raster, **vrt_opts)

    def _load_dataset(self):
        dataset = []

        # For each city find the asscociated rasters
        for city in self.city_list:
            # For each raster open the geotif, extract the bounds and store this with the file path and city
            for raster_src in glob(f"{self.opt_dir}/{city}/PSM*/*_3035.tif"):
                raster = self._create_virtual_warped_raster( rasterio.open(raster_src), resolution=self.resolution )
                dataset.append({
                    "sensor": "OPT",
                    "city": city,
                    "path": os.path.relpath(raster_src, self.base_dir),
                    "geometry": self._create_bounding_box(raster),
                    "raster": raster
                })

            #  Do the same for the SAR data
            for raster_src in glob(f"{self.sar_dir}/{city}/dims*/*/*/IMAGEDATA/*_3035.tif"):
                raster = self._create_virtual_warped_raster( rasterio.open(raster_src), resolution=self.resolution )
                dataset.append({
                    "sensor": "SAR",
                    "city": city,
                    "path": os.path.relpath(raster_src, self.base_dir),
                    "geometry": self._create_bounding_box(raster),
                    "raster": raster
                })

        if len(dataset) > 0:
            df = pd.DataFrame.from_dict(dataset)
            return gpd.GeoDataFrame(df, geometry="geometry", crs=self.crs.srs)

    # Function reads geojson file corresponding to the optical image and loads the geometry for each city
    # Returns a dictionary of geometry/feature points indexed by city name
    # load_existing will try and load preprocessed points such that we do not need to call reduce on the dataset again
    def _load_geometry(self, cities=None, load_existing=True):
        city_list = cities if cities is not None else self.city_list
        fpts = {}
        existing = True

        for city in city_list:
            processed_pnts_src = os.path.join(self.opt_dir, city, f"{city}_geometry.geojson")
            if load_existing and os.path.exists(processed_pnts_src):
                fpts[city] = gpd.read_file(processed_pnts_src)

                if "wkt" not in fpts[city]:
                    fpts[city]["wkt"] = fpts[city]["geometry"].apply(lambda x: str(x))
            else:
                for gjson_src in glob(f"{self.opt_dir}/{city}/PSM*/*_3035_{self.scheme}.geojson"):
                    data = self.dataset[ self.dataset.path == os.path.relpath(gjson_src, self.base_dir).replace(f"_{self.scheme}.geojson", ".tif") ]

                    # Only load points if there is a corresponding raster for the geojson
                    if len(data) > 0:
                        points_df = gpd.read_file(gjson_src)

                        # To allow for grouping by points
                        points_df["wkt"] = points_df["geometry"].apply(lambda x: str(x))

                        # If the geometry is empty then skip it
                        if len(points_df) > 0:
                            fpts[city] = points_df
                            existing = False

        return fpts, existing

    def _load_windows(self, cities=None):
        city_list = cities if cities is not None else self.city_list
        windows = {}

        def to_window(w):
            w = ast.literal_eval(w)
            return Window.from_slices(*w)

        for city in city_list:
            processed_windows_src = os.path.join(self.opt_dir, city, f"{city}_windows.geojson")
            if os.path.exists(processed_windows_src):
                ws = gpd.read_file(processed_windows_src)
                ws["window"] = ws.window.apply(to_window)
                ws = ws.merge(self.dataset.loc[:, ["path", "city", "sensor", "raster"]], left_on="path", right_on="path")
                ws = ws.merge(self.geometry[city].loc[:, ["wkt", "size", "response"]], left_on=["wkt"], right_on=["wkt"], how="left")
                windows[city] = ws

        return windows

    # Only keep points which occur in all rasters in the reduction set
    # Keep extended keeps the information relating to the path and valid rasters, otherwise we only keep the point set
    def _reduce_geometry(self, geometry, city, reduction_set, keep_extended=False):
        search = {}
        for (_, item) in self.dataset.loc[self.dataset.city == city, ["geometry", "sensor", "path"]].iterrows():
            within = geometry.within(item.geometry)
            if item.sensor not in search:
                search[item.sensor] = within
            else:
                search[item.sensor] = within | search[item.sensor]

            search[item.path] = within

        # Create a new lookup table mapping points to their rasters
        selection = geometry.assign(**search).melt(id_vars=["geometry", "wkt", "size", "response", "SAR", "OPT"], var_name="path", value_name="valid")

        if reduction_set is None:
            reduction_set = ["valid"]
        else:
            reduction_set = reduction_set + ["valid",]

        selection["valid"] = selection[reduction_set].apply(all, axis=1)
        selection = selection.loc[selection.valid == True]
        selection.drop(columns=["valid", "SAR", "OPT"], inplace=True)

        if not keep_extended:
            selection = selection.drop(columns=["path"]).drop_duplicates(subset="wkt")

        return selection

    # Returns only geometry, if reduce is true then only returns points which exist in the reduction set
    # If load_exising is true then no reduce operation will be run if it succeeds in loading the existing dataset
    # This property is mainly meant for loading the dataset initially, it will have no effect once geometry has already been loaded
    def get_geometry(self, cities=None, reduce=False, reduction_set=["SAR", "OPT"], load_existing=True):
        geometry = self.geometry
        city_list = cities if cities is not None else self.city_list
        reduction_set = reduction_set
        cached = True

        if geometry is None:
            geometry, existing = self._load_geometry(cities=city_list, load_existing=load_existing)
            cached = cached and existing
        else:
            for city in city_list:
                if city not in geometry or geometry[city] is None:
                    geom, existing = self._load_geometry(cities=[city], load_existing=False)

                    if geom and len(geom) > 0:
                        geometry[city] = geom
                        cached = cached and existing

        # Don't reduce if we loaded this from the cached points we saved before - we can assume they are already reduced
        if not reduce:
            reduction_set = None

        # Ensure overlap between loaded geometry and city keys first as a geometry might have been empty
        city_list = set(city_list).intersection(set(geometry.keys()))

        for city in city_list:
            if reduce and not cached:
                # Only keep point information
                geometry[city] = self._reduce_geometry(geometry[city], city, reduction_set=reduction_set, keep_extended=False)

        geometry = {k: v for k, v in geometry.items() if k in city_list}

        return geometry

    # For each city in the geometry list save
    def save_geometry(self):
        assert self.geometry is not None, "Geometry of your Urban Atlas class is empty, nothing to save"

        for city in self.geometry.keys():
            fname = os.path.join(self.opt_dir, city, f"{city}_geometry.geojson")

            try:
                os.remove(fname)
            except:
                pass
            finally:
                if len(self.geometry[city]) > 0:
                    self.geometry[city].to_file(fname, driver="GeoJSON")

    def save_windows(self, windows):
        for city in windows.keys():
            if len(windows[city]) > 0:
                # Only keep the WKT of the point, the raster path and the Window. The remaining information can be recovered
                # From a join to the point file, or to the dataset file.
                w = windows[city].drop(columns=["city", "sensor", "raster", "size", "response"])
                w["window"] = w.window.apply(lambda x: str(x.toranges()))
                fname = os.path.join(self.opt_dir, city, f"{city}_windows.geojson")

                try:
                    os.remove(fname)
                except:
                    pass
                finally:
                    if len(w) > 0:
                        w.to_file(fname, driver="GeoJSON")

    def _create_patch_window(self, row, size_px=256):
        point = row.geometry
        raster = row.raster

        row, col = raster.index(point.x, point.y)

        # Find the top left corner
        row = row - size_px//2
        col = col - size_px//2

        window = Window(col, row, size_px, size_px)
        window = rasterio.windows.get_data_window(raster).intersection(window)

        if rasterio.windows.shape(window) != (size_px, size_px):
            return None

        return window

    # Returns a list of rasterio windows centered on each point.
    # If reduce is true then we ensure that the patches are valid and exist in both modalities
    # else we return all patches, even if they contain nodata values
    # The None values will not be removed from the list, to ensure that the windows can be aligned with the original point list
    # NOTE: The check_nodata of windows takes a long time due to the need to read the raster data to determine if the patch is full
    # If using this for training it is suggested to rather do this check on the fly.
    def get_windows(self, points, size_px=256, check_nodata=False, reduce=False, reduction_set=["SAR", "OPT"], load_existing=False):
        def check_valid(grp):
            grp['valid'] = set(grp.sensor.values) == set(reduction_set)
            return grp

        windows = {}
        for city in points.keys():
            if load_existing:
                joint = self._load_windows(cities=[city])
                windows.update(joint)
                # TODO: Filter to only the provided points
            else:
                # Ensure points are fully loaded and intialised before we try create windows
                if "path" not in points[city].columns:
                    raster_point_lut = self._reduce_geometry(points[city], city, reduction_set=reduction_set, keep_extended=True)

                joint = raster_point_lut.merge(self.dataset.loc[:, ["path", "city", "sensor", "raster"]], left_on="path", right_on="path")
                joint["window"] = joint.apply(lambda r: self._create_patch_window(r, size_px=size_px), axis=1)

                if reduce:
                    joint = joint[joint.window.notnull()]
                    joint = joint.groupby("wkt").apply(check_valid)
                    joint = joint[joint.valid]
                    joint.drop(columns=["valid"], inplace=True)
                    # As the point list would now be reduced due to removing invalid windows
                    # points[city] = joint.drop(columns=["path", "sensor", "raster", "window"]).drop_duplicates(subset="wkt")
                    # Filter the points to only those which have valid windows
                    points[city] = points[city].loc[points[city].wkt.isin(joint.wkt.unique())]

                windows[city] = joint

        return windows

    # Note this function takes the window from the get_windows function, not a raw rasterio window.
    def get_patch(self, window_df, masked=True, transform=None):
        try:
            patch = window_df.raster.read(window=window_df.window, masked=True)
            
            if masked and patch.mask.any():
                return None
        except:
            return None

        return transform(patch) if transform is not None else patch

    # Close all open file handles
    def close(self):
        self.dataset.apply(lambda r: r.raster.close(), axis=1)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tqdm import tqdm

    parser = ArgumentParser()
    parser.add_argument('-o', '--optsrc', default="/run/user/1000/gvfs/smb-share:server=sipeo-nas.local,share=nas/Datasets/UrbanAtlas/PRISM", help="Source directory for optical images")
    parser.add_argument('-s', '--sarsrc', default="/run/user/1000/gvfs/smb-share:server=sipeo-nas.local,share=nas/Datasets/UrbanAtlas/TSX", help="Source directory for optical images")
    parser.add_argument('-c', '--cities', nargs='*', help="List of cities to load, not specifying will load all cities")
    args = parser.parse_args()

    # ua = UrbanAtlas("/media/lloyd/Seagate Expansion Drive/Datasets/Urban Atlas/UrbanAtlas(Opt)/", "/media/lloyd/Seagate Expansion Drive/Datasets/Urban Atlas/TerraSAR-X/", cities=["Bonn", "Braga"])
    with MeasureDuration() as m:
        ua = UrbanAtlas(args.optsrc, args.sarsrc, cities=args.cities, crs="EPSG:3035", load_geometry=True, workers=None, stats=False, force=True, scheme="HARRIS")

    with MeasureDuration() as m:
        points = ua.get_geometry(cities=args.cities, reduce=True, load_existing=True)

    windows = {}
    with MeasureDuration() as m:
        windows = ua.get_windows(points, reduce=True)

    # ua.save_geometry()
    # ua.save_windows(windows)

    with MeasureDuration() as m:
        windows = ua.get_windows(points, reduce=True, load_existing=True)

    import code
    code.interact(local=locals())