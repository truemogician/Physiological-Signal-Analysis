## Environment

The environment of this project is managed by [conda](https://docs.conda.io/en/latest/). To get the environment ready, here's the steps:

1. Install [conda](https://docs.conda.io/en/latest/) if you haven't.
2. Add conda to your system path.
3. Run the following command from project root: `conda env create -f environment.yml -n {env_name}`. Replace `{env_name}` with the name you want to give to the environment. This environment takes 8.02GB on my Windows 11 machine, so make sure you have enough space.
4. To execute scripts in this project, run `conda activate {env_name}` first to activate the environment.

## Dataset

Currently, this research project uses [WAY-EEG-GAL](https://figshare.com/collections/WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376) only.

To get the data ready, here's the steps:

1. Download the dataset from figshare. This dataset includes 12 sub-datasets, one for each participant. The sizes of those sub-datasets vary from 700MB to 1000MB, here's the basic information and download page of them. You don't need to downloa all of them, but if I recommend downloading at least 2 of them, so that you can test the code on different participants.
   |participant|size| download page |
   |:-:| :-: | :-: |
   | 1 | 799.88MB | [P1.zip](https://figshare.com/articles/dataset/Participant_1_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1185502)|
   | 2 | 905.53MB | [P2.zip](https://figshare.com/articles/dataset/Participant_2_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1185505) |
   | 3 | 709.43MB | [P3.zip](https://figshare.com/articles/dataset/Participant_3_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1185507) |
   | 4 | 764.96MB | [P4.zip](https://figshare.com/articles/dataset/Participant_4_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1185509) |
   | 5 | 803.37MB | [P5.zip](https://figshare.com/articles/dataset/Participant_5_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1185511) |
   | 6 | 822.52MB | [P6.zip](https://figshare.com/articles/dataset/Participant_2_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119392) |
   | 7 | 988.70MB | [P7.zip](https://figshare.com/articles/dataset/Participant_7_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119691) |
   | 8 | 745.41MB | [P8.zip](https://figshare.com/articles/dataset/Participant_8_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119669) |
   | 9 | 793.71MB | [P9.zip](https://figshare.com/articles/dataset/Participant_9_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119677) |
   | 10 | 812.92MB | [P10.zip](https://figshare.com/articles/dataset/Participant_10_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119682) |
   | 11 | 847.04MB | [P11.zip](https://figshare.com/articles/dataset/Participant_11_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119680) |
   | 12 | 866.80MB | [P12.zip](https://figshare.com/articles/dataset/Participant_12_in_the_WAY_EEG_GAL_dataset_328_grasp_and_lift_trials_with_different_weights_and_surfaces_during_which_EEG_EMG_kinematics_and_kinetics_were_recorded/1119678) |
2. Move the downloaded zip files into `data/WAY-EEG-GAL` folder.
3. Run the following command from project root: `python script/dump_all.py data/WAY-EEG-GAL`. If you want to preserve the zip files for whatever reason, you can add `--no-delete` option to the command. Note that this script relies on the original file names, so if you want it to work, don't rename the zip files.
4. After the command succeeds, the data is now ready to be used. The structure of the data folder should looks like following:
    ```
    data
    └── WAY-EEG-GAL
        ├── sub-01
        │   ├── raw
        │   │   ├── HS_P1_S1.mat
        │   │   ├── ...
        │   │   ├── HS_P1_S9.mat
        │   │   ├── WS_P1_S1.mat
        │   │   ├── ...
        │   │   └── WS_P1_S9.mat
        │   ├── series-01
        │   │   ├── metadata.json
        │   │   └── samples.npz
        │   ├── series-02
        │   │   ...
        │   └── series-09
        ├── ...
        └── sub-12
    ```
    The `raw` folder contains the original mat files extracted from the zip files. These MatLab files are not further used in this project, so you may delete them at your will. You'll save approximately 75% of the disk space by doing so.
