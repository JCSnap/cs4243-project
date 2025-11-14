## SOC Compute stuff

1. login into xxx@xlogin.comp.nus.edu.sg
2. scp to copy the train npy and csv file from [drive](https://drive.google.com/drive/folders/1G40aRVz98yQiqm689cn7cQStrwSs7eHK?usp=drive_link) and test file (non segmented) from [drive](https://drive.google.com/drive/folders/1kvo1VmSdn6t-cPX8cjOTOsbJdd_9wOt5?usp=drive_link), zip the test file before copying into remote and unzipping. You may also have to change the train and test file path in `train.py` accordingly
3. install relevant python packages by running `setup.sh`
4. train the model by running `train_model.sh`, you might have to change the path if you are not running on the home directory
5. test the model with `test_model.sh`
6. use scp to copy the model weights and test result to local
