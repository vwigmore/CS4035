Steps for executing code with .tar.xz
--
1. Create folder 'datasets' with all data sets from .tar.xz file

  `Optionally - Run visualisation.py`
  
  `Optionally - Run smote_analysis.py`
  
  `Optionally - Run blackbox.py`

Steps for executing code without .tar.xz
--
1. Create empty folder 'datasets'

  `Optionally - Run visualisation.py`
  
2. Import encoding.knwf into knime and save result in dataset folder

  `Optionally - Run smote_analysis.py`
  
3. Run preprocessing.py
4. Import smote_for_kfold.knwf into knime and save result in dataset folder

  `Optionally - Run blackbox.py`
