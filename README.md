# AI-Powered Waste Classification for Smart Recycling

## Project Overview

This project develops a deep learning-based image classification system to identify different types of waste and support smarter recycling processes.

Two deep learning approaches are implemented and compared:

- Custom Convolutional Neural Network (CNN)
- MobileNetV2 using Transfer Learning

The objective is to evaluate both approaches and identify the model that provides better classification performance across nine waste categories.

## Waste Categories

The system classifies images into nine categories:

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

It contains 4,752 images across nine categories.

| Category | Images |
|---|---:|
| Cardboard | 461 |
| Food Organics | 411 |
| Glass | 420 |
| Metal | 790 |
| Miscellaneous Trash | 495 |
| Paper | 500 |
| Plastic | 921 |
| Textile Trash | 318 |
| Vegetation | 436 |
| **Total** | **4,752** |

The dataset is divided using stratified sampling:

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 3,326 | 70% |
| Validation | 713 | 15% |
| Testing | 713 | 15% |

The test set is kept separate during model development and used for final evaluation on unseen images.

Download the dataset and place the nine category folders inside:

`data/Waste_Dataset/`

The raw dataset is not included in this GitHub repository.

## Data Preprocessing

The preprocessing pipeline includes:

- Resizing images to 224 × 224 pixels.
- Converting images to RGB format.
- Encoding the nine waste categories into numerical labels.
- Stratified splitting into training, validation, and testing sets.
- Normalizing Custom CNN images to the range [0, 1].
- Applying MobileNetV2 preprocessing to scale image values to [-1, 1].
- Applying data augmentation to the MobileNetV2 training pipeline, including horizontal flipping, rotation, and zoom.

## Models

### 1. Custom CNN

A CNN architecture developed specifically for this project.

It consists of:

- Three convolutional layers with 32, 64, and 128 filters.
- Max-pooling layers.
- Global Average Pooling.
- A Dense layer with 128 neurons and ReLU activation.
- Dropout with a rate of 0.3.
- A final Softmax layer with nine output classes.

### 2. MobileNetV2 Transfer Learning

MobileNetV2 uses pretrained ImageNet weights as a feature extractor.

The classification head consists of:

- Data augmentation.
- Global Average Pooling.
- Dropout with a rate of 0.3.
- A final Dense layer with nine Softmax outputs.

The pretrained base model is initially frozen during training.

## Technologies and Libraries

- Python 3.11
- TensorFlow 2.20.0
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- IPython Kernel

## Experimental Results

Both models were trained for 50 epochs using the same dataset split and evaluated on the validation and independent test sets.

### Training and Validation Results

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Epochs Completed | 50 | 50 |
| Training Accuracy | 67.29% | 80.37% |
| Training Loss | 0.8929 | 0.5768 |
| Validation Accuracy | 69.00% | 76.44% |
| Validation Loss | 0.8963 | 0.6708 |

### Independent Test Results

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 65.92% | 75.46% |
| Loss | 0.9127 | 0.6493 |
| Macro Precision | 0.67 | 0.76 |
| Macro Recall | 0.64 | 0.76 |
| Macro F1-score | 0.65 | 0.76 |
| Weighted F1-score | 0.65 | 0.75 |

MobileNetV2 achieved 9.54 percentage points higher test accuracy than the Custom CNN.

It also achieved a higher F1-score across all nine waste categories, demonstrating stronger overall classification performance.

### Final Model Selection

MobileNetV2 was selected as the final model because it demonstrated better overall classification performance, lower loss, and stronger generalization to unseen images.

The confusion matrices for both models are available in `results/figures/`.

## Prediction Pipeline

A prediction pipeline was implemented in `predict.py` using the trained MobileNetV2 model.

It allows users to:

- Provide an external JPG, JPEG, or PNG image.
- Automatically resize and preprocess the image.
- Predict one of the nine waste categories.
- View the predicted category and confidence score.
- View the top three prediction results.

The pipeline was tested using two additional unseen images:

- `metal_test.png` – correctly classified as Metal.
- `plastic_hard_test.png` – correctly classified as Plastic.

An additional Jupyter Notebook, `evaluate.ipynb`, was created to evaluate these external predictions against their actual labels and display accuracy, a classification report, and a confusion matrix.

These two images are additional functional tests and are separate from the original 713-image independent test set. They should not be interpreted as a representative evaluation of overall model performance.

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
│   ├── custom_cnn_best.keras
│   └── mobilenetv2_best.keras
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_data_preprocessing.ipynb
├── results/
│   ├── figures/
│   │   ├── custom_cnn_confusion_matrix.png
│   │   └── mobilenetv2_confusion_matrix.png
│   └── metrics/
├── predict.py
├── evaluate.ipynb
├── metal_test.png
├── plastic_hard_test.png
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

1. Install Python 3.11.
2. Clone or download this repository.
3. Open the project folder in VS Code.
4. Create and activate a virtual environment.
5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

6. Register the Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name waste-classification --display-name "Python (Waste Classification)"
```

7. Download and extract the dataset into `data/Waste_Dataset/`.
8. Open the notebooks in the `notebooks/` folder and select the appropriate Python kernel.

### Running the Prediction Pipeline

To classify a new image, run:

```bash
python predict.py metal_test.png
```

Replace `metal_test.png` with the path to your own JPG, JPEG, or PNG image.

The trained model must be available at `models/mobilenetv2_best.keras`.

## Current Development Status

- Dataset exploration completed.
- Image dimensions and quality inspected.
- Data preprocessing pipeline implemented.
- Custom CNN architecture implemented and trained for 50 epochs.
- MobileNetV2 transfer learning implemented and trained for 50 epochs.
- Validation and independent test evaluation completed.
- Updated classification reports generated.
- Confusion matrices generated.
- Model comparison completed using the latest experimental results.
- MobileNetV2 selected as the final model.
- Image prediction pipeline implemented in `predict.py`.
- External image prediction successfully tested.
- Prediction confidence and top-three results implemented.
- Evaluation notebook created in `evaluate.ipynb`.

## Future Improvements

- Fine-tune selected MobileNetV2 layers.
- Improve classification of challenging waste categories, particularly Miscellaneous Trash.
- Explore additional data augmentation techniques.
- Evaluate performance using a larger and more diverse dataset.

## Author

**Zad Mahmoud Naanoush**