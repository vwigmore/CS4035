Steps for executing code with datasets already preprocessed
--
1. Create folder 'datasets' with all data sets from datasets_part1.zip, datasets_part2.zip and datasets_part3.zip

  `Optionally - Run visualisation.py`
  
  `Optionally - Run smote_analysis.py`
  
  `Optionally - Run blackbox.py`

  `Optionally - Run whitebox.py`

Steps for executing code without preprocessed datasets
--
1. Create empty folder 'datasets'

  `Optionally - Run visualisation.py`
  
2. Import encoding.knwf into knime and save result in dataset folder

  `Optionally - Run smote_analysis.py`
  
3. Run preprocessing.py	
4. Import smote_for_kfold.knwf into knime and save result in dataset folder
5. Move header.txt into folder datasets

  `Optionally - Run blackbox.py`

  `Optionally - Run whitebox.py`
