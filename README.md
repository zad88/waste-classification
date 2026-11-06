# AI-Powered Waste Classification for Smart Recycling

## Project Overview

This project develops an image classification system using Deep Learning to identify different types of waste and support smart recycling.

The system uses Convolutional Neural Networks (CNNs) to classify waste images into nine categories.

## Waste Categories

1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

## Dataset

The dataset is obtained from Kaggle:

[Waste Classification Dataset](https://www.kaggle.com/datasets/adithyachalla/waste-classification/data)

The dataset contains 4,752 images across nine categories.

Download the dataset and place the nine category folders inside:

`data/Waste_Dataset/`

The dataset is not included in this GitHub repository.

## Technologies and Libraries

- Python 3.11
- TensorFlow 2.20
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
waste-classification/
├── data/
│   └── Waste_Dataset/
├── docs/
│   ├── development_log.md
│   ├── learning_notes.md
│   └── experiment_log.csv
├── models/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_data_preprocessing.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── report/
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

1. Install Python 3.11.
2. Clone or download this repository.
3. Open the project folder in VS Code.
4. Create and activate a virtual environment.
5. Install the required libraries:

```bash
pip install -r requirements.txt
```

6. Install and register the Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name waste-classification --display-name "Python (Waste Classification)"
```

7. Open the notebooks in the `notebooks` folder and select the appropriate Python kernel.

## Current Development Progress

- Dataset exploration completed.
- Image dimensions and quality inspected.
- Training and validation datasets prepared.
- Image preprocessing pipeline implemented.
- Custom CNN architecture created and compiled.
- Model training and evaluation are in progress.

## Future Work

- Train and evaluate the custom CNN.
- Implement a transfer learning model.
- Compare model performance.
- Evaluate classification metrics and confusion matrices.
- Develop an image prediction pipeline.
- Prepare the final project report.

## Author

Zad Mahmoud Naanoush