# 3D Head Reconstruction Pipeline

A hybrid classical/neural pipeline that reconstructs a watertight 3D head mesh from a set of RGB images. Runs fully offline on a single consumer GPU in under 5 minutes.

<p align="center">
  <table align="center">
    <tr>
      <td align="center"><img src="assets/front.png" width="270" alt="Front view"/></td>
      <td align="center"><img src="assets/side.png" width="270" alt="Side view"/></td>
      <td align="center"><img src="assets/back.png" width="270" alt="Back view"/></td>
    </tr>
    <tr>
      <td align="center"><em>Front view.</em></td>
      <td align="center"><em>Side view.</em></td>
      <td align="center"><em>Back view.</em></td>
    </tr>
  </table>
</p>

## Approach

The pipeline combines MODNet (PyTorch) for foreground segmentation with COLMAP for camera pose estimation and dense multi-view stereo. The resulting point cloud is processed through Poisson Surface Reconstruction to generate a closed mesh. A geometry-based post-processing stage automatically detects the shoulder boundary, caps the open base, fills remaining holes, and decimates to the target face count. The final mesh is exported with vertex normals and corrected orientation.

## Dependencies overview

- **PyTorch** — MODNet inference (GPU-accelerated foreground segmentation).
- **COLMAP** — Structure-from-Motion and Multi-View Stereo.
- **Open3D** — Poisson Surface Reconstruction and mesh processing.
- **PyMeshLab** — Hole filling and mesh repair.

## Quick Start

```bash
python run.py --input ./images --output mesh.ply
```

## Setup

### 1. System dependencies

```bash
sudo apt install colmap
```
### 2. Recommended installation (fast and reliable)
```bash
conda create -n head_3d_reconstruction python=3.10 -y
conda activate head_3d_reconstruction

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install Pillow numpy open3d pymeshlab requests pysocks gdown
```

### 3. Alternative (environment file)
```bash
conda env create -f environment.yml 
conda activate head_3d_reconstruction
```

### 4. MODNet weights

```bash
pip install gdown
gdown 1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX -O modnet.ckpt
```

Weights source: [MODNet pretrained on portrait matting](https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX)

### 5. MODNet source

```bash
git clone https://github.com/ZHKKKe/MODNet modnet_src
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| --input | required | Directory of input images |
| --output | required | Output path (.ply) |
| --quality | medium | Reconstruction quality: low, medium, high |
| --max-faces | 50000 | Max faces after decimation |
| --no-gpu | false | Disable GPU |
| --weights | ./modnet.ckpt | Path to MODNet weights |
| --work-dir | ./workdir | Intermediate files directory |

## Benchmarks

Measured on RTX 3080 with 100 images (frames extracted from video):

| Stage | Time |
|---|---|
| Preprocessing | ~0.2s |
| Masking (MODNet) | ~2.5s |
| SfM (COLMAP) | ~15s |
| Dense MVS (COLMAP) | ~155s |
| Poisson meshing | ~13s |
| Post-processing | ~0.5s |
| Total | ~186s |

## Output

A .ply file with up to 50K triangular faces, vertex normals, vertex colors, corrected orientation (Y-up, face-forward).

## Pipeline

| Stage | Module | Description |
|---|---|---|
| Preprocessing | pipeline/preprocessing.py | RGBA to RGB conversion |
| Masking | pipeline/masking.py | MODNet foreground segmentation |
| SfM | pipeline/sfm.py | COLMAP sparse reconstruction |
| Dense | pipeline/dense.py | COLMAP patch-match stereo and fusion |
| Meshing | pipeline/meshing.py | Poisson surface reconstruction |
| Post-processing | pipeline/postprocess.py | Shoulder detection, capping, decimation, export |
