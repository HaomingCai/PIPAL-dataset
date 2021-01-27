


## General Data Process
- We only use random crop for data augmentation. 
- Remember reset the normalization based on your need.
- We use opencv (`cv2`) to read and process images.


## How To Prepare Training Data
1. BAPPS 2afc dataset. Download using [BAPPS offical script](https://github.com/richzhang/PerceptualSimilarity/blob/master/scripts/download_dataset.sh).
> Modify configurations in `codes/options/train_test_yml/train_our_IQA.yml` when training, e.g., `train_root`, `valid_root` and `train_valid` (Choose which part of BAPPS 2afc dataset become training data).

2. PIPAL dataset. Download [Training](https://drive.google.com/drive/folders/1G4fLeDcq6uQQmYdkjYUHhzyel4Pz81p-), [Validation](https://drive.google.com/drive/folders/1w0wFYHj8iQ8FgA9-YaKZLq7HAtykckXn) and [Testing (2021-03-01)]().
> For PIPAL training dataset, you could obtain four files named Distortion_1, Distortion_2, Distortion_3 and Distortion_4. Please create one file which only contains these four subfiles.
<br/> Modify configurations in `codes/options/train_test_yml/train_our_IQA.yml` when training, e.g., `mos_root`, `ref_root` and `dis_root`.

> For PIPAL validation (Ref_valid / Distortion_valid), place them in the one file. 
<br/> When you want to test your model. Modify configurations in `codes/options/train_test_yml/test_IQA.yml` when testing, e.g., `ref_root` and `dis_root`.

3. The ideal file structure is shown below.
> Important! The "Distortion" file should only contains four subfiles described here. Otherwise, you should modify function [image_combinations](data_util.py) in [codes/data/data_util.py](data_util.py).

```  
PIPAL
   │   
   └───validation (Public)
   │   |
   │   └───Reference_valid
   │   │     |A0000.bmp
   │   │     |A0003.bmp
   │   │     |...
   │   │            
   │   └───Distortion_valid
   │         |A0000_10_00.bmp
   │         |A0000_10_01.bmp
   │         |...
   │   
   └───training (Public)
       │
       └───MOS_Scores_train
       │     |A0001.txt
       │     |A0002.txt
       │     |...
       │ 
       └───Reference_train
       │     |A0000.bmp
       │     |A0003.bmp
       │     |...
       │            
       └───Distortion 
             └───Distortion_1
             │       | ...
             └───Distortion_2
             │       | ...
             └───Distortion_3
             │       | ...
             └───Distortion_4
                     | ...
