import numpy as np 
import os
import shutil
import hashlib

from glob import glob

# A basic folder cache for Numpy objects
class BasicCache:
    # type = "lifo", "fifo", "fill"
    def __init__(self, cache_dir, size=10000, scheme="fill", clear=True, overwrite=False):
        self.basedir = os.path.abspath(cache_dir)
        self.size = size
        self.scheme = scheme
        self.overwrite = overwrite
        self.cache = {}
        self.times = []

        if clear:
            self.clear()

        os.makedirs(self.basedir, exist_ok=True)
        self._load_existing_cache()

    def key_hash(self, key):
        return hashlib.sha224(key.encode()).hexdigest()

    def _load_existing_cache(self):
        # We only cache numpy objects
        flist = glob(os.path.join(self.basedir, "*.npz"))

        for f in flist:
            h = os.path.splitext(os.path.basename(f))[0]

            if len(self.cache) < self.size:
                self.cache[h] = f
                self.times.append(h)

    def _get_filename(self, key):
        h = self.key_hash(key)
        return os.path.join(self.basedir, "{}.npz".format(h)), h

    def _prune_cache(self):
        # Remove an item from the cache according to scheme
        if len(self.cache) >= self.size:
            if self.scheme == "lifo":
                rm_idx = self.times.pop()
            elif self.scheme == "fifo":
                rm_idx = self.times.pop(0)
            else:
                return False

            rm_file = self.cache[rm_idx]
            # Remove the cached file
            if os.path.exists(rm_file):
                os.remove(rm_file)
    
            del self.cache[rm_idx]
        return True

    def clear(self):
        if os.path.exists(self.basedir):
            shutil.rmtree(self.basedir)

    def isin(self, key):
        _, h = self._get_filename(key)
        return h in self.times

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, key):
        _, h = self._get_filename(key)

        if self.isin(key):
            fname = self.cache[h]
            data = np.load(fname, allow_pickle=True)
            
            if "arr_0" in data.files:
                return data["arr_0"]
            else:
                return {f: data[f] for f in data.files}

    def __setitem__(self, key, value):
        fname, h = self._get_filename(key)

        # Only add the item if it isn't already in the cache
        if (self.overwrite and h in self.times) or self._prune_cache():
            if isinstance(value, dict):
                np.savez(fname, **value)
            else:
                np.savez(fname, value)

            self.cache[h] = fname

            if h not in self.times:
                self.times.append(h)

if __name__=="__main__":
    cache = BasicCache("tmp_cache", size=100, scheme="fill", clear=True)

    # Create 120 random objects and cache them (only 100 should cache)
    for i in range(0, 120):
        cache[i] = np.random.rand(4, 4)

    for i in range(0, 120):
        data = cache[i]

        if data is not None:
            print(f"Retrieved {i} = {data.shape} form cache")
        else:
            print(f"{i} not in cache")

    cache2 = BasicCache("tmp_cache", size=100, scheme="fill", clear=False, overwrite=True)
    cache[45] = np.random.rand(10, 10)
    print(f"Retrieved without overwrite {cache[45].shape}")
    cache2[45] = np.random.rand(10, 10)
    print(f"Retrieved with overwrite {cache2[45].shape}")
