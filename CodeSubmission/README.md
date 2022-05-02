# CS310 Project - James Page

### This folder contains the code associated with my project document. 

### There are two main folders:
- Data
- Development

## Data
This folder contains all data that is hard accessed by the code, including the main training data set, the locations list, the statistical distributions of weather events for each location and the precomputed predictions of generation outputs for each location.

This also contains the Data collection notebook which queries APIs to add entries to the dataset. This feeds each set of results into the "archived" folder also present in this folder, which stores individual runs of queries so they can be added to the dataset.

Finally the results folder stores the graph trace images of the optimisers search progress.

## Development
Within the development folder there is the TPOT_search script, and another subfolder containing the final model and the genetic algorithm code. This also contains the PredictorModel notebook which was used in our development of our prediction model up til the use of the tpot_search file.

This structure should be kept or if files are moved, the file paths within the scripts should also be moved to account for the new location (code curently uses relative paths).

To run the allocationOptimiser script, it can be ran in python with no arguments and it will use a default starting seed, however if you want to search with a specified random seed, you can pass the seed in as and argument:
```bash
python allocationOptimser.py insert_seed_here
```

This folder also contains the Allocations.txt file which stores all the top allocations of a run along with its objective values, and the parameters it was run with.

### Libraries

Some python libraries may need installation before running the code.

The requirements for each script/file will be seen by what libraries are imported, however the main libraries used are:

- Pandas
- NumPy
- SkLearn (installation instructions -> https://scikit-learn.org/stable/install.html)
- Seaborn & Matplotlib
- Xgboost (installation instructions -> https://xgboost.readthedocs.io/en/stable/install.html)
- TPOT (installation instructions -> http://epistasislab.github.io/tpot/installing/)

Most libraries can be installed using python, however others that require more prerequisites have more instructions provided above.

```bash
python pip install library_name
```

