---
title: Download
layout: page
---

<div class="hero has-text-centered" id="download">
<div class="myWrapper" markdown="1" align="left">

## Download (API)

The Eyecandies dataset can be used with [Pipelime](https://github.com/eyecan-ai/pipelime-python), an open-source python framework that provides tools to help you build automated data pipelines.

Please refer to the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo for more info and examples on how to use and download the dataset using the **Pipelime API**.

## Download (Manual)

If you don't want to use Pipelime, you can manually download the dataset from the following links:

- [Candy Cane](https://drive.google.com/file/d/1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY/view?usp=sharing)
- [Chocolate Cookie](https://drive.google.com/file/d/1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN/view?usp=sharing)
- [Chocolate Praline](https://drive.google.com/file/d/1dRlDAS31QJSwROgA6yFcXo85mL0EBh25/view?usp=sharing)
- [Confetto](https://drive.google.com/file/d/10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq/view?usp=sharing)
- [Gummy Bear](https://drive.google.com/file/d/1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W/view?usp=sharing)
- [Hazelnut Truffle](https://drive.google.com/file/d/1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm/view?usp=sharing)
- [Licorice Sandwich](https://drive.google.com/file/d/1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC/view?usp=sharing)
- [Lollipop](https://drive.google.com/file/d/1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz/view?usp=sharing)
- [Marshmallow](https://drive.google.com/file/d/1pebIU3AegEFilqqoROaVzOZqkSgX-JTo/view?usp=sharing)
- [Peppermint Candy](https://drive.google.com/file/d/1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF/view?usp=sharing)

</div>
</div>

# FORMAT

<div class="hero has-text-centered" id="format">
<div class="myWrapper" markdown="1" align="left">

Inside each object class you will find the following:
<!-- | --------------- | ------------------------------------------------------------------ | ---------------- | --------------- | -->

| **Folder Name** | **Description**                                                    | **Good Samples** | **Bad Samples** |
| `train`         | The training set, fit your model on this data.                     | 1000             | 0               |
| `val`           | The validation set, use this data to tune hyper parameters.        | 100              | 0               |
| `test_public`   | The **public part** of the test set with ground-truth annotations. | 25               | 25              |
| `test_private`  | The **private part** of the test set without any annotation.       | 200              | 200             |

Each dataset sample contains the following items:
<!-- | -------------- | --------------------------------------------------------- | --------------- | ------------ | -->

| **Item**       | **Description**                                           | **Format**      | **Public**\* |
| `image_0`      | RGB image with all spotlights on                          | PNG RGB 24-bit  | ✔️            |
| `image_1`      | RGB image with bottom spotlight on                        | PNG RGB 24-bit  | ✔️            |
| `image_2`      | RGB image with right spotlight on                         | PNG RGB 24-bit  | ✔️            |
| `image_3`      | RGB image with top spotlight on                           | PNG RGB 24-bit  | ✔️            |
| `image_4`      | RGB image with left spotlight on                          | PNG RGB 24-bit  | ✔️            |
| `image_5`      | RGB image with global lighting                            | PNG RGB 24-bit  | ✔️            |
| `depth`        | Depth map normalized between 0 and 65535                  | PNG GRAY 16-bit | ✔️            |
| `info_depth`   | Depth map normalization min/max bounds                    | YAML            | ✔️            |
| `normals`      | Normals map relative to the camera                        | PNG RGB 24-bit  | ✔️            |
| `pose`         | Camera 4x4 pose matrix relative to the world              | TXT             | ✔️            |
| `obj_params`   | Geometry and shading parameters for both object and scene | YAML            | ❌            |
| `metadata`     | Image-level anomaly classification labels                 | YAML            | ❌            |
| `bumps_mask`   | GT segmentation mask for bump-type anomalies              | PNG GRAY 8-bit  | ❌            |
| `dents_mask`   | GT segmentation mask for dent-type anomalies              | PNG GRAY 8-bit  | ❌            |
| `colors_mask`  | GT segmentation mask for color-type anomalies             | PNG GRAY 8-bit  | ❌            |
| `normals_mask` | GT segmentation mask for normal-type anomalies            | PNG GRAY 8-bit  | ❌            |
| `mask`         | Pixel-wise or between all previous GT masks               | PNG GRAY 8-bit  | ❌            |

\* Only "public" items are available in the private test set. Still, they are available for the other sets.

# Depth Map De-Normalization

To get the depth map in meters, you need to de-normalize it using the `info_depth` file.
Here a sample code using plain python:

```python
import yaml
import imageio.v3 as iio
import numpy as np


def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg


depth_meters = load_and_convert_depth("path/to/depth.png", "path/to/info_depth.yaml")
```

The [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo provides a ready-to-use **[Pipelime](https://github.com/eyecan-ai/pipelime-python) stage** to perform the conversion on-the-fly.

# Normals Map recovery

Normals are saved as RGB images where each pixel maps the `(nx, ny, nz)` normal vector from `[-1, 1]` float to `[0, 255]` uint8 `(red, green, blue)`.
Also, the reference frame has Z and Y flipped with respect to the camera reference frame,
so you should account for it before moving to the world reference frame.
Here a sample code in python:

```python
import imageio.v3 as iio
import numpy as np


def load_and_convert_normals(normal_img, pose_txt):
    # input pose
    pose = np.loadtxt(pose_txt)

    # input normals
    normals = iio.imread(normal_img).astype(float)
    img_shape = normals.shape

    # [0, 255] -> [-1, 1] and normalize
    normals = normalize(normals / 127.5 - 1.0, norm="l2")

    # flatten, flip Z and Y, then apply the pose
    normals = normals.reshape(-1, 3) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    normals = normals @ pose[:3, :3].T

    # back to image, if needed
    normals = normals.reshape(img_shape)
    return normals


normals = load_and_convert_normals("path/to/normals.png", "path/to/pose.txt")
```

The [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo provides a ready-to-use **[Pipelime](https://github.com/eyecan-ai/pipelime-python) stage** to compute the normals and the pointcloud.

# Depth Map To Pointcloud Conversion

A basic conversion from Depth to Pointcloud can be done by defining the camera projection matrix as in the following plain python snippet:

```python
import numpy as np

def depth_to_pointcloud(depth_img, info_depth, pose_txt, focal_length):
    # input depth map (in meters) --- cfr previous section
    depth_mt = load_and_convert_depth(depth_img, info_depth)

    # input pose
    pose = np.loadtxt(pose_txt)

    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose

    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    # invert and apply to each 4-vector
    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T

    # remove the homogeneous coordinate
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    return pcd[:, :3]


# The same camera has been used for all the images
FOCAL_LENGTH = 711.11

pc = depth_to_pointcloud(
    "path/to/depth.png",
    "path/to/info_depth.yaml",
    "path/to/pose.txt",
    FOCAL_LENGTH,
)
```

To directly create a ply pointcloud with points, colors and normals,
we also provide a **stage** `depth2pc` in the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo.
For example, the following snippet shows how to build a simple CLI to compute a show a metric pointcloud with open3d:

```python
import typer
import numpy as np
import open3d as o3d
from pathlib import Path

from pipelime.sequences import SamplesSequence

from eyecandies.stages import DepthToMetersStage, DepthToPCStage


def main(
    dataset_path: Path = typer.Option(..., help="Eyecandies Dataset"),
):
    # Load the dataset
    seq = SamplesSequence.from_underfolder(dataset_path)

    # Apply the stages
    seq = seq.map(DepthToMetersStage())
    seq = seq.map(DepthToPCStage())

    # setup the open3d visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True

    for sample in seq:
        # get the pointcloud as trimesh object
        pcd = sample["pcd"]()

        # converting to open3d pointcloud
        pcd = pcd.as_open3d()

        # scale for better visualization
        pcd = pcd.scale(10, np.array([0.0, 0.0, 0.0]))

        # show the pointcloud
        vis.add_geometry(pcd)
        vis.run()
        vis.remove_geometry(pcd)

    vis.destroy_window()

if __name__ == "__main__":
    typer.run(main)
```

Upon launching the above command you shall be able to visualize your colored pointclouds:

![Alt text](assets\images\pc\point_cloud.gif "Candy Cane point cloud visualization")

# Conversion To Anomalib Data Format

Do you want to use the data within the [Anomalib](https://github.com/openvinotoolkit/anomalib) framework? Checkout the [Eyecandies](https://github.com/eyecan-ai/eyecandies) repo to find a ready-to-use converter!

</div>
</div>
