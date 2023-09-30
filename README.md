# Panoramic Localization
Panoramic localization library containing PyTorch implementations of various panoramic localization algorithms: PICCOLO (ICCV 2021), CPO (ECCV 2022), and [LDL](https://github.com/82magnolia/panoramic-localization#running-ldl) (ICCV 2023).

## Dataset preparation (Stanford 2D-3D-S & OmniScenes)
First ownload the panorama images (`pano`) and poses (`pose`) from the following [link](https://docs.google.com/forms/d/e/1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform?c=0&w=1) (download the one without `XYZ`) and the point cloud (`pcd_not_aligned`) from the following [link](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1).
Also, download the 3D line segments through the following [link](https://drive.google.com/file/d/1Ur67nN8Q2n_CXQxbI341TRUbQEmtEjuD/view?usp=sharing).
Then, place the data in the directory structure below.

    piccolo/data
    └── stanford (Stanford2D-3D-S Dataset)
        ├── pano (panorama images)
        │   ├── area_1
        │   │  └── *.png
        │   ⋮
        │   │
        │   └── area_6
        │       └── *.png
        ├── pcd (point cloud data)
        │   ├── area_1
        │   │   └── *.txt
        │   ⋮
        │   │
        │   └── area_6
        │       └── *.txt
        ├── pcd_line (line cloud data)
        │   ├── area_1
        │   │   └── *.txt
        │   ⋮
        │   │
        │   └── area_6
        │       └── *.txt
        └── pose (json files containing ground truth camera pose)
            ├── area_1
            │   └── *.json
            ⋮
            │
            └── area_6
                └── *.json

To obtain results in OmniScenes, please refer to the download instructions [below](https://github.com/82magnolia/piccolo#downloading-omniscenes-update-new-scenes-added).
Note we are using the **old** version of OmniScenes for this repository.
In addition, download the 3D line segments through the following [link](https://drive.google.com/file/d/1M7A5iDXQdrPVUNmhKWRSFMQKit07jOK8/view?usp=sharing).
Then, place the data in the directory structure below.

    piccolo/data
    └── omniscenes (OmniScenes Dataset)
        ├── change_handheld_pano (panorama images)
        │   ├── handheld_pyebaekRoom_1_scene_2 (scene folder)
        │   │  └── *.jpg
        │   ⋮
        │   │
        │   └── handheld_weddingHall_1_scene_2 (scene folder)
        │       └── *.jpg
        └── change_handheld_pose (json files containing ground truth camera pose)
        |   ├── handheld_pyebaekRoom_1_scene_2 (scene folder)
        |   │   └── *.json
        |   ⋮
        |   │
        |   └── handheld_pyebaekRoom_1_scene_2 (scene folder)
        |       └── *.json
        ⋮
        └── pcd_line (line cloud data)
        |   ├── pyebaekRoom_1lines.txt
        |   │
        |   ⋮
        |   │
        |   └── weddingHall_1lines.txt
        └── pcd (point cloud data)
            ├── pyebaekRoom_1.txt
            │
            ⋮
            │
            └── weddingHall_1.txt

## Installation
To run the codebase, you need [Anaconda](https://www.anaconda.com/). Once you have Anaconda installed, run the following command to create a conda environment.

    conda create --name omniloc python=3.7
    conda activate omniloc
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html 
    conda install cudatoolkit=10.1

In addition, you must install pytorch_scatter. Follow the instructions provided in [the pytorch_scatter github repo](https://github.com/rusty1s/pytorch_scatter). You need to install the version for torch 1.7.0 and CUDA 10.1.

## Running LDL
### Stanford 2D-3D-S or OmniScenes
Run the following command for Stanford 2D-3D-S.
```
python main.py --config config/stanford_ldl.ini --log log/ldl_test --method ldl
```

Similarly, run the following command for OmniScenes
```
python main.py --config config/omniscenes_ldl.ini --log log/ldl_test --method ldl
```

### Preparing and Testing on Your Own Data
We also provide scripts for directly testing on your own data. 
First, prepare a query panorama image and 3D colored point cloud.
Then, extract 3D line segments using the [following repository](https://github.com/xiaohulugo/3DLineDetection).
Finally, run the following command, and the renderings at the localized pose will be saved in the log directory.
```
python main.py --config config/omniscenes_ldl.ini --log LOG_DIRECTORY --method ldl --single --query_img PATH_TO_QUERY_IMG --color_pcd PATH_TO_COLORED_POINT_CLOUD --line_pcd PATH_TO_LINE_CLOUD
```
