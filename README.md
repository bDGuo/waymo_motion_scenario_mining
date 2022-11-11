# waymo_motion_scenerio_mining

## Introduction

This is a .

## Scenario Classification

This repo contains a python implementation of scenario classification for the dataset [Waymo Motion](https://waymo.com/open/data/motion).

## Dataset

The tfexample from [Waymo Motion](https://waymo.com/open/data/motion) is organized by time-frame.
Download dataset in shell:

```shell
gcloud init # not needed after the first time
```

```shell
gcloud auth login # go to browser and login with google account
```

```shell
gsutil config # config project name
```

```shell
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example" \
  "your_data_dir"
```
