
1. Setup and activate Python environment:
```
conda create --name msynth_env python=3.9.13
conda activate msynth_env
pip install numpy==1.23.3
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install pytorch-lightning==1.7.7
pip install opencv-python==4.6.0.66
pip install matplotlib==3.6.0
pip install timm==0.8.17.dev0
pip install torchmetrics==0.10.2
pip install scikit-image==0.19.3
pip install pandas==1.5.0
pip install scikit-learn==1.1.2
pip install huggingface_hub==0.18.0
pip install jupyter==1.0.0
```

Set up huggingface hub token for data download:
- Follow instructions on [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) to make a token. 
- Run and paste your token:
```
huggingface-cli login
```


2. **Preprocessing and Training**. In order to train a model on the M-synth dataset, first check the ```preprocess_and_train_device_models.py``` script: 

* Modify ```dir_head``` in ```util/config.py``` to point to location of the root directory 
* _preprocess_images_ set to True if the data requires pre-processing. 
* _split_train_test_ set to True if the data needs to be split into train, validation and test datasets. For each subset of the data, two subfolders of "nolesion" and "withlesion" will be created that separates the images based on the presence of the lesion. The lesion's presence will be determined based on the log files created during the phantom generation. 
* _train_classifier_ is set to True if the model needs to be trained on the M-synth data. 
* The script divides 300 available examples (images) for each phantom with specific parameters into 200 images for training, 50 images for validation and 50 images for test. 
* The default model in "efficientnet-b0" can be changed in ```util/util_classifier.py``` script.

The preprocessing code can be run via:
```
cd code
python -u preprocess_and_train_device_models.py --density fatty --detector SIM --size 5.0 --lesiondensity 1.06 --dose 2.22e10
python -u preprocess_and_train_device_models.py --density fatty --detector SIM --size 7.0 --lesiondensity 1.06 --dose 2.22e10
python -u preprocess_and_train_device_models.py --density fatty --detector SIM --size 9.0 --lesiondensity 1.06 --dose 2.22e10
```
(OPTIONAL): if you want to train the model on the provided data, run:
```
python -u preprocess_and_train_device_models.py --density fatty --detector SIM --size 5.0 --lesiondensity 1.06 --dose 2.22e10 --train
```


3. **Testing and Evaluation**.

Either train the models, or download the pretrained models using the provided script:

```
python -u download_setup_pretrained_models.py --density fatty --detector SIM --size 7.0 --lesiondensity 1.06 --dose 2.22e10
```

The ```testing.py``` script will create a dictionary of the test results. The dataset split and stored in the test subfolder will be used for test. 

```
for LESION_TEST_SIZE in 5.0 7.0 9.0 ; do 
    python -u testing.py --density fatty --detector SIM --size 7.0 --lesiondensity 1.06 --dose 2.22e10 --density_test fatty --size_test $LESION_TEST_SIZE --lesiondensity_test 1.06 --dose_test 2.22e10
done 
```		
	
4. In order to visualize the results of the inference step, use generate_figures.ipynb script which will plot the results based on the values stored in the dictionaries during the test step.
Note that you need Rscript to run iMRMC. Please run the following:
```
sudo apt install r-base-core
cd imrmc/
R
```
type the following lines, making sure to save workspace image in R:
```
install.packages('iMRMChongfei_0.0.0.9000.tar.gz',type='source',repos=NULL)
install.packages('iMRMC_1.2.4.tar.gz',type='source',repos=NULL)
quit()
```
then run notebook:
```
cd ../
jupyter notebook generate_figures.ipynb
```

** Tips **

Conversion table between percent dose and value (# of histories) can be found below:

| Density/Percent Dose | 20%     | 40%     | 60%     | 80%     | 100%    |
|----------------------|---------|---------|---------|---------|---------|
| Dense                | 1.73e09 | 3.47e09 | 5.20e09 | 6.94e09 | 8.67e09 |
| Hetero               | 2.04e09 | 4.08e09 | 6.12e09 | 8.16e09 | 1.02e10 |
| Scattered            | 4.08e09 | 8.16e09 | 1.22e10 | 1.63e10 | 2.04e10 |
| Fatty                | 4.44e09 | 8.88e09 | 1.33e10 | 1.78e10 | 2.22e10 |




