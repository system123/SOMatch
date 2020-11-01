import rasterio
import cv2
import os
import geojson
import zipfile
import tarfile
import pyproj

import numpy as np
import geopandas as gpd
import pandas as pd
import skimage as ski

from skimage.feature import peak_local_max, corner_peaks, corner_harris
from argparse import ArgumentParser
from rasterio.windows import get_data_window
from geojson import Feature, Point, FeatureCollection
from tqdm import tqdm

from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform
from rasterio.enums import Resampling

class FakeCVFpt:
    def __init__(self, xy):
        self.size = 1
        self.response = 1
        self.pt = (xy[1], xy[0])

def create_virtual_warped_raster(raster, projection="EPSG:3035", resolution=2.5):
        epsg_code = int(projection.split(':')[1])

        # Broken PRISM images
        if raster.crs is None:
            raster._crs = CRS.from_epsg(epsg_code)

        if raster.crs is not None and raster.crs.to_epsg() == epsg_code:
            return raster
        
        dst_transform, dst_width, dst_height = calculate_default_transform(raster.crs, projection, raster.width, raster.height, *raster.bounds, resolution=(resolution, resolution))
        vrt_opts = {
            'resampling': Resampling.nearest,
            'crs': projection,
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            }

        return WarpedVRT(raster, **vrt_opts)

def load_from_geotif(src, band, roi=None):
    if roi is None:
        roi = get_data_window(src.read(band, masked=True))

    img = src.read(band, window=roi)
    width = src.width
    height = src.height
    transform = src.transform

    return img, (width, height), transform

def find_keypoints(img, scheme="SURF", radius=None):
    if scheme == "SURF":
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=3, extended=False, upright=True)
    elif scheme == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3, sigma=1.3)
    elif scheme == "BRISK":
        detector = cv2.BRISK_create(thresh=30, octaves=3)
    elif scheme == "ORB":
        detector = cv2.ORB_create(nfeatures=10000)

    if scheme not in ["HARRIS"]:
        kps = detector.detect(img, None)
    else:
        cnrs = corner_peaks(corner_harris(img), min_distance=radius)
        kps = [FakeCVFpt(xy) for xy in cnrs]

    return kps

def superimpose_keypoints(img, fpts):
    img2 = cv2.drawKeypoints(img, fpts, None, (255,0,0), 4)
    return img2

# pseudo non-maximal suppression of the keypoints based on selecting the maximal point within a radius of r
def keypoints_nms(img, keypoints, r=64):
    feature_map = np.zeros_like(img).astype(np.float64)

    for img_pt, kpt in keypoints.items():
        feature_map[img_pt] = kpt["response"]

    # Find the peaks in the original feature map and ensure they are seperated by at least r+1 pixels
    coords = corner_peaks(feature_map, min_distance=r, exclude_border=True)

    # Rebuild the keypoint list with all features from the NMS selected features
    keypoint_list = {}
    for pt in coords:
        keypoint_list[tuple(pt)] = keypoints[tuple(pt)]

    return keypoint_list

# Convert openCV keypoints to world cordinates
def cv_keypoints_to_world(kpts, tif, epsg="EPSG:4326"):
    keypoints = {}

    for kp in tqdm(kpts):
        x, y = kp.pt
        (lon, lat) = tif.xy(y, x)

        img_pt = (int(y), int(x))

        if img_pt not in keypoints or keypoints[img_pt]["response"] < kp.response:
            keypoints[img_pt] = {"pt_w": (lon, lat), "pt_i": (int(x), int(y)), "size": kp.size, "response": kp.response}

            if epsg is not None:
                dest_proj = pyproj.Proj(init=epsg)
                pt_wgs84 = pyproj.transform(tif.crs.to_proj4(), dest_proj, lon, lat)
                keypoints[img_pt][epsg] = (pt_wgs84[0], pt_wgs84[1])

    return keypoints

def keypoints_to_geojson(keypoints, geom="pt_w"):
    point_list = []

    for _, kpt in keypoints.items():
        point_list.append( Feature( geometry=Point(kpt[geom]), properties=kpt ) )

    return FeatureCollection(point_list)

def open_raster(src, temp_dest=None, name_prefix=None):
    _, ext = os.path.splitext(src)
    raster = None
    fmem = None

    if ext == ".tif":
        raster = create_virtual_warped_raster( rasterio.open(src) )
    elif ext == ".zip":
        archive = zipfile.ZipFile(src, 'r')
        tifs = [fname for fname in archive.infolist() if os.path.splitext(fname.filename)[1] == '.tif']

        if len(tifs) > 0:
            if temp_dest is None:
                f = archive.open(tifs[0], 'r')
                fmem = io.MemoryFile(f.read())
                raster = fmem.open()
            else:
                tifs[0].filename = os.path.basename(tifs[0].filename)
                if name_prefix is not None:
                    tifs[0].filename = "{}_{}".format(name_prefix, tifs[0].filename)

                archive.extract(tifs[0], path=temp_dest)
                raster = rasterio.open(os.path.join(temp_dest, tifs[0].filename))

    elif ext in ['.tar', '.gz', '.bz2', '.xz']:
        archive = tarfile.TarFile(src, 'r')
        tifs = []
        for member in archive.getmembers():
            # TSX Data archive contains numerous tif files so ensure we only open the actual image data
            if 'IMAGEDATA' in member.path and os.path.splitext(member.path)[1] == '.tif':
                tifs.append(member)

        if len(tifs) > 0:
            if temp_dest is None:
                f = archive.extractfile(tifs[0])
                fmem = io.MemoryFile(f.read())
                raster = fmem.open()
            else:
                tifs[0].name = os.path.basename(tifs[0].name)
                if name_prefix is not None:
                    tifs[0].name = "{}_{}".format(name_prefix, tifs[0].name)

                archive.extract(tifs[0], path=temp_dest)
                raster = rasterio.open(os.path.join(temp_dest, tifs[0].filename))

    return raster, fmem

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src", type=str, help="GeoTif to find feature points in")
    parser.add_argument("-o", "--output", action="store_true", help="Save features to CSV file and as geojson")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot feature points")
    parser.add_argument("-b", "--band", type=int, default=1, help="Band in which to detect feature points")
    parser.add_argument("-r", "--radius", type=int, default=63, help="Radius for non-maximal suppression of keypoints")
    parser.add_argument("-s", "--scheme", type=str, default="SURF", help="Feature point detection scheme")
    parser.add_argument("-c", "--cut_patches", type=int, default=-1, help="Cut patches of specified size centered around the detected feature point")
    parser.add_argument("-g", "--geometry", type=str, default="pt_w", help="The geometry to use in the geojson object. 'pt_w' is raster projection, 'pt_i' is the image coords, 'epsg' the proj specified with -e is used.")
    parser.add_argument("-e", "--epsg", type=str, default=None, help="Specify an epsg code to reproject all points to. By default no reprojection happens")

    # parser.add_argument("-roi", "--roi", nargs=4, default=[0, 0, -1, -1], help="Geo-coords of bounding box in which to find feature points")
    args = parser.parse_args()

    # args.src = "/Volumes/Hades/Documents/Varsity/PhD_Remote_Sensing/Data and Experiments/Athens/PSM_MMC_TP__0002631001.317.1/O.tif"

    assert os.path.exists(args.src), "File does not exist"

    if args.epsg is not None:
        args.epsg = "EPSG:{}".format(args.epsg)

        if args.geometry == 'epsg':
            args.geometry = args.epsg

    print(args.src)
    
    tif, tmp_file = open_raster(args.src)
    img, dims, transform = load_from_geotif(tif, 1)
    print(img.shape)

    kpts = find_keypoints(img, scheme=args.scheme, radius=args.radius)
    keypoint_list = cv_keypoints_to_world(kpts, tif, epsg=args.epsg)
    print(f"Found {len(keypoint_list)} keypoints")

    if args.plot:
        img2 = superimpose_keypoints(img, kpts)

        meta = tif.meta.copy()
        meta.update({
            "count": img2.shape[-1]
            })

        assert len(img2.shape) == 3, "Incorrect image size, expected (H, W, C)"

        fname = os.path.splitext(args.src)
        fname = "{}_keypoints.{}".format(*fname)

        with rasterio.open(fname, 'w', **meta) as dst:
            for i in range(img2.shape[-1]):
                dst.write(img2[:,:,i], i+1)

    # Supress non-maximal keypoints
    keypoint_list = keypoints_nms(img, keypoint_list, r=args.radius)
    print(f"{len(keypoint_list)} keypoints after NMS")

    featcol = keypoints_to_geojson(keypoint_list, geom=args.geometry)
    
    df = pd.DataFrame.from_dict(list(keypoint_list.values()))

    if args.output:
        fname = os.path.splitext(args.src)
        df.to_csv( "{}_{}.csv".format(fname[0], args.scheme) )

        print(f"Created GeoJSON with {len(featcol['features'])} keypoints")

        filename = "{}_{}.geojson".format(fname[0], args.scheme)

        try:
            os.remove(filename)
        except:
            pass
        finally:
            with open(filename, 'w') as dest:
                geojson.dump(featcol, dest)

    print(len(keypoint_list))