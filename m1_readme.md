# Project Documentation

## Project Overview
This project explores the performance of various machine learning models, including Fully Connected Neural Networks (FCN), Support Vector Machines (SVM), Logistic Regressions, on concatenated  MNIST datasets. It includes hyperparameter tuning, model evaluation, performance comparison, and concludes with t-SNE visualization for feature representation analysis.

## Folder and File Structure
Below is the tree structure of the project:

```plaintext
project/
├── balanced_models/
│   ├── model_trial_bal_1.h5
│   ├── ...
│   └── model_trial_bal_20.h5
├── best_models/
│   ├── best_fcn_hyperparameters.json
│   ├── best_fcn_model_balanced.h5
│   ├── best_fcn_model_imbalanced.h5
│   └── svm_rbf_model.joblib
├── fcn_tuning_bal/
│   ├── mnist_addition_classification_bal/
│   │   ├── trial_00/
│   │   │   ├── checkpoint_temp
│   │   │   ├── checkpoint
│   │   │   ├── checkpoint.data-00000-of-00001
│   │   │   ├── checkpoint.index
│   │   │   └── trial.json
│   │   ├── ...
│   │   └── trial_19/
├── imbalanced_models/
│   ├── model_trial_inbal_1.h5
│   ├── ...
│   └── model_trial_inbal_20.h5
├── performance_results/
│   ├── classification_report_method1_20000.csv
│   ├── classification_report_method2_20000.csv
│   ├── classification_report.csv
│   ├── evaluation_results.json
│   ├── random_forests_results.json
│   ├── svm_report.csv
│   └── svm_trail_performance.json
├── Report/
│   └── M1_Coursework_Report.pdf
├── .gitignore
├── Coursework.ipynb
├── fcn_performance_compare.ipynb
├── LICENSE
├── M1_Coursework.pdf
├── readme.md
└── requirements.txt

```

### Folder Descriptions

#### 1. `balanced_models/`
This folder contains `.h5` files for the 20 models trained during hyperparameter tuning on the **balanced dataset**.

#### 2. `best_models/`
This folder stores the best-performing models and hyperparameters:
- `best_fcn_hyperparameters.json`: JSON file containing the best hyperparameters for the FCN.
- `best_fcn_model_balanced.h5`: Best model trained on the balanced dataset.
- `best_fcn_model_imbalanced.h5`: Best model trained on the imbalanced dataset.
- `svm_rbf_model.joblib`: SVM model file.

#### 3. `fcn_tuning_bal/`
This folder contains the Keras Tuner-generated engineering files for hyperparameter tuning on the balanced dataset. 

#### 4. `imbalanced_models/`
This folder contains `.h5` files for the 20 models trained using the same hyperparameters as the balanced dataset but trained on the **imbalanced dataset**. 

#### 5. `performance_results/`
This folder contains performance metrics for different models.

#### 6. `Report/`
Contains the project reports in PDF format

#### 7. `Coursework.ipynb`
The main Jupyter notebook containing the code for the project.

#### 8. `fcn_performance_compare.ipynb`
This notebook compares the performance of balanced and imbalanced models on both balanced and imbalanced test sets.

#### 9. `requirements.txt`
List of Python dependencies required for this project. It ensures that the environment can be reproduced.



## How to Set Up the Environment

1. **Using Conda**:

   - Navigate to the directory containing `requirements.txt`:
     ```bash
     cd <path to requirements.txt>
     ```

   - Create a virtual environment:
     ```bash
     conda env create -n <venv_name> python=3.9 -y
     ```

   - Activate the virtual environment:
     ```bash
     conda activate <venv_name>
     ```

   - Install the dependencies:
     ```bash
     pip install --no-cache-dir -r requirements.txt
     ```

2. **Select the Environment in Your IDE**:
   - Choose `<venv_name>` as the active environment in your IDE (e.g., VS Code, PyCharm, JupyterLab).



## Notes
- Some cells in the Jupyter notebooks may start with `WARNING` because they require a long runtime. For these cells, the results have been saved and the computation cell has been commented out. The next cell will load the results for convenience.



## Declaration of Auto Generation Tools

This project leverages AI tools to assist in the development process. Specifically:
- **Code**: Portions of the code were generated using ChatGPT-4o based on pseudocode and instructions provided by the author.
- **Report**: The project report and documentation were created using ChatGPT-4o, guided by the author's detailed instructions.

However, all ideas, concepts, and the overall project structure are entirely the author's own.


## License
This project is licensed under the terms specified in the `LICENSE` file.
