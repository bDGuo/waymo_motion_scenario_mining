# Scenario Extraction from a Large Real-World Dataset

This repo contains a python implementation of scenario extraction from a large real-world dataset, i.e., [Waymo Open Motion Dataset](https://waymo.com/open/data/motion).

## Dataset

Before running the code, please download the dataset v1.1 in tf_example format files.

## Running the code

1. Install the packages in virtual environment with [requirements.txt](requirements.txt). The code is tested with Python 3.8.10.
2. To generate the tags, run with the following command.

   ```shell
   python .\utils\runner.py --data_dir your_data_path
   # An example of your_data_path is "./waymo_v_1_1/tf_example/training"
   ```

   The tags will be stored in an automatically created folder which is named by the time you run the code in [./results/](./results/)
3. To categorize the scenarios by searching for a combination of tags, run the following command

   ```shell
   python .\utils\runner_sc.py --result_time your_result_time
   # MM-DD-HH_MM
   ```

## Extracted scenarios

We have extracted a total of 215,090 scenarios across three scenario categories, which is accessible [here](https://drive.google.com/file/d/1i7kxqBIosCxKK2SI_cP6srOqkre_iDkc/view?usp=share_link).
