from json import load
import numpy as np
from plyfile import PlyData, PlyElement

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def convert_ply2bin(ply_path,bin_path=None):
    plydata = PlyData.read(ply_path)
    vertex =plydata['vertex']
    x,y,z,i= vertex['x'],vertex['y'],vertex['z'],vertex['intensity']/65535
    pcd = np.stack([x,y,z,i],axis=1)
    if bin_path:
        pcd.tofile(bin_path)
    return pcd

if __name__ == "__main__":
    bin_pcd = load_from_bin('./example/000104.bin')
    ply_pcd = convert_ply2bin('./example/frame000104-1581624663_170.ply')
    print(np.sum(bin_pcd-ply_pcd))