# Subgroup Fairness Assessment

## Description
This project aims to assess the fairness of machine learning models in terms of subgroup predictive bias. It provides tools and metrics to evaluate the performance of models across different subgroups and identify potential biases based on model loss explanations.

## Requirements
To run the project, you need to have Python 3.7 or higher installed (3.8 recommended). You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Run
To run the project, you can use the following command:
```bash
python app.py
```

To see all available options, you can use the CLI:
```bash
python app.py --help
```

# Loading own dataset
To load your own dataset, please edit the `import_dataset` function in load.py, retaining the same function signature (returning X and y_true).
