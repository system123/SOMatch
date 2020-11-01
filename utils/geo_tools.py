import rasterio
import numpy as np

from itertools import product
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.csgraph import csgraph_from_dense, depth_first_tree

from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.plot import show, show_hist

def bbox_intersection(bounds1, bounds2):
    if disjoint_bounds(bounds1, bounds2):
        raise Exception("Bounds are disjoint, no interseciton exists")
    
    bbox = BoundingBox(
                left=max(bounds1.left, bounds2.left),
                right=min(bounds1.right, bounds2.right),
                top=min(bounds1.top, bounds2.top),
                bottom=max(bounds1.bottom, bounds2.bottom)
            )

    return bbox

def relative_window(base_window, abs_window, strict=True):
    window = Window(
        col_off=base_window.col_off + abs_window.col_off,
        row_off=base_window.row_off + abs_window.row_off,
        width=abs_window.width,
        height=abs_window.height,
        )

    return window.intersection(base_window) if strict else window

def absolute_window(base_window, rel_window, strict=True):
    window = Window(
        col_off=rel_window.col_off - base_window.col_off,
        row_off=rel_window.row_off - base_window.row_off,
        width=rel_window.width,
        height=rel_window.height,
        )

    return window.intersection(base_window) if strict else window

class Raster:
    def __init__(self, src_path, bands=None):
        self.path = src_path
        self.raster = rasterio.open(self.path)
        self.bands = bands

        # These will change when doing set operations
        self.width = self.raster.width
        self.height = self.raster.height
        self.window = Window(0, 0, self.width, self.height)
        self.transform = self.raster.transform
        self.profile = self.raster.profile.copy()

    def _update_profile(self):
        self.profile.update({
            'height': self.height,
            'width': self.width,
            'transform': self.transform
        })

    # Assumes that bounds are in the same CRS
    def _clip_bounds(self, bounds):
        if disjoint_bounds(bounds, self.raster.bounds):
            raise Exception("Bounds are disjoint, no interseciton exists")

        # Get the new bounds as a window in the original raster
        bounds_window = rasterio.windows.from_bounds(*bounds, transform=self.raster.transform)
        bounds_window = bounds_window.intersection(self.window)

        self.window = bounds_window.round_lengths(op='ceil')
        self.height = int(self.window.height)
        self.width = int(self.window.width)
        self.transform = rasterio.windows.transform(self.window, self.transform)

        self._update_profile()
        
    def clip_bounds_by_raster(self, template, intersection=False):
        bounds = template.raster.bounds

        if template.raster.crs != self.raster.crs:
            # Make sure bounds are in same coordinate system
            bounds = transform_bounds(template.raster.crs, self.raster.crs, *bounds)

        if intersection:
            bounds = bbox_intersection(self.raster.bounds, bounds)

        self._clip_bounds(bounds)

    # Same as clip by raster except only valida data regions in both images are kept
    def crop_by_raster(self, template):
        pass

    # Takes in a set of patch offsets (relative) and outputs windows for each which generates a full sized patch (if strict)
    def _get_patch_windows_from_offsets(self, offsets, size):
        abs_window = Window(0, 0, self.width, self.height)

        for col_off, row_off in  offsets:
            window = Window(col_off=col_off, row_off=row_off, width=size, height=size).intersection(abs_window)

            if (window.width != size or window.height != size):
                continue

            transform = rasterio.windows.transform(relative_window(self.window, window, strict=True), self.transform)
            yield window, transform

    # If strict is false then all points will be returned, if stride is specified then it will be enforced at best effort (guarentee no larger overlap, but don't guarentee all points)
    # Points must be a numpy array of Mx2 (cols, rows) or (x, y), it is assumed points represent the center pixels of the patch
    def get_patch_windows_from_imgXY(self, pts, size, stride=None):
        # Convert to top left corner first
        pts = pts - (size//2)

        idxs = np.all(pts >= 0, axis=1)
        idxs = np.logical_and(np.logical_and( np.all(pts >= 0, axis=1), pts[:, 0] <= self.width ),
                                pts[:, 1] <= self.height)
        pts = pts[idxs, :]

        if stride:
            selected_pts = np.empty(shape=(0, 2))

            # Sort by x - we'll iterate that way first
            sort = np.argsort(pts[:,0])
            pts = pts[sort, :]
            
            for pt in pts:
                pt = pt[np.newaxis,:]
                if len(selected_pts) > 0:
                    valid = np.all(cdist(pt, selected_pts) >= stride)
                else:
                    valid = True

                if valid:
                    selected_pts = np.concatenate([selected_pts, pt], axis=0)

            pts = selected_pts

        return self._get_patch_windows_from_offsets(pts, size)

    # Using geocoords rather than raster image coords
    def get_patch_windows_from_worldXY(self, pts, size, stride=None):
        r, c = rasterio.transform.rowcol(self.transform, xs=pts[:,0], ys=pts[:,1])
        ptsXY = np.stack([c, r], axis=1)

        # Clean up invalid points - negative, or outside of the main data window
        # This isn't needed but is more efficient than processing points which we know
        # won't be selected
        idxs = np.all(ptsXY >= 0, axis=1)
        idxs = np.logical_and(np.logical_and( np.all(ptsXY >= 0, axis=1), ptsXY[:, 0] <= self.width ),
                                ptsXY[:, 1] <= self.height)
        ptsXY = ptsXY[idxs, :]
                
        return self.get_patch_windows_from_imgXY(ptsXY, size, stride=stride)
  
    # Get patch windows with size and stride, only full windows will be returned
    def get_patch_windows(self, size, stride):
        offsets = product(range(0, self.width, stride), range(0, self.height, stride))
        return self._get_patch_windows_from_offsets(offsets, size)

    def _get_patches(self, bands, windows, strict=True):
        for window, transform in windows:
            data = self.read(bands, window=window, masked=True)

            # If the patch contains nodata values then don't generate it
            if strict and np.ma.is_masked(data):
                continue
            
            yield data, transform

    # Yields patches from the raster with the defined stride and size, strict means no nodata will exist in thee returned patches
    def get_patches(self, bands=1, size=128, stride=64, strict=True):
        return self._get_patches(bands, self.get_patch_windows(size, stride), strict=strict)

    def get_patches_from_imgXY(self, bands):
        pass

    def read(self, bands=None, window=None, masked=False):
        if window is None:
            window = self.window
        else:
            window = relative_window(self.window, window)

        return self.raster.read(bands, window=window, masked=masked)

    def save(self, dest):
        with rasterio.open(dest, "w", **self.profile) as dest:
            dest.write(self.read(out_shape=(self.raster.count, self.height, self.width)))

    def show(self, band=1, cmap="terrain", window=None, **kwargs):        
        show((self.read(band, window=window)), cmap=cmap, transform=self.transform, **kwargs)

    def show_hist(self, bands=1, bins=50, **kwargs):
        show_hist(self.read(bands), bins=bins, **kwargs)

if __name__=="__main__":
    a=Raster("/media/Zambezi/Data/RawGeoTifs/Munich_WV2.tif")
    b=Raster("/media/Zambezi/SimGeoI_Data/GeoTif/TSX_Frauenkirche_Munich.tif")
    pts = np.load("/media/Zambezi/SAR_OPT_Data/Munich_Center_17112017/coords.npy")
    a.clip_bounds_by_raster(b)
    # a.show(window=Window(0,0,128,128), cmap="gray")
    # b.show_hist(bins=10, alpha=0.3, lw=0.0, histtype='stepfilled', title='Histogram', masked=True)
    print(a.window)
    print(a.transform)
    # for w,t in b.get_patch_windows(128, 128, strict=True):
    #     print(f"{w}")

    # for i, (patch, t) in enumerate(a.get_patches(1, size=256, stride=128, strict=True)):
    #     show(patch, transform=t, cmap="gray", title=i)
    import code
    code.interact(local=locals())