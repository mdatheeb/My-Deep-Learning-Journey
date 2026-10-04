# Breast Cancer Classification with a Neural Network (TensorFlow/Keras)

A multilayer neural network that classifies breast-mass samples as **malignant** or **benign** using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

This project is a learning exercise in applying a neural-network classification workflow to a real dataset (Deep Learning, Experiment 2). It is **not** a medical diagnosis tool.

## Dataset

- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC), UCI Machine Learning Repository
- **Files:** `wdbc.data` (the data) and `wdbc.names` (the description)
- **Size:** 569 samples (357 benign, 212 malignant)
- **Features:** 30 numeric measurements computed from digitized images of fine needle aspirates of breast masses
- **Label:** diagnosis, `M` = malignant, `B` = benign

The 30 features come from 10 cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each recorded as a **mean**, **standard error** and **worst** (largest) value.

`wdbc.data` has no header row. Its columns are: ID, diagnosis, then the 30 features.

## Pipeline

```
Load data -> Preprocess -> Train/Test split -> Feature scaling
   -> Input (30) -> Dense (ReLU) -> Dense (ReLU) -> Output (Sigmoid)
   -> Compile -> Train -> Evaluate -> Predict
```

| Step | What it does |
|------|--------------|
| Preprocessing | Drops the ID column and encodes the label (malignant = 1, benign = 0) |
| Split | 80% train / 20% test, stratified so both sets keep the same class ratio |
| Scaling | `StandardScaler` fitted on the training set only, then applied to test and new samples to avoid data leakage |
| Model | Input (30) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(1, Sigmoid) |
| Compile | Adam optimizer, binary cross-entropy loss, accuracy metric |
| Training | 50 epochs, batch size 16, 20% of the training data held out for validation |
| Evaluation | Test loss and accuracy, confusion matrix, precision/recall/F1 per class |
| Prediction | Scales new samples with the same fitted scaler and outputs P(malignant) |

The output layer has one sigmoid neuron that gives the probability of the malignant class. Probabilities above 0.5 are predicted as malignant.

## Project structure

```
.
├── breast_cancer_nn_wdbc.py   # full pipeline script
├── wdbc.data                  # dataset
├── wdbc.names                 # dataset description
└── README.md
```

## Setup

Python 3.11 or 3.12 is recommended, since TensorFlow can lag behind the newest Python releases.

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   ```

3. Install the dependencies:

   ```
   pip install tensorflow scikit-learn pandas matplotlib
   ```

## Usage

Make sure `wdbc.data` is in the same folder as the script (or update `DATA_PATH` at the top of the script), then run:

```
python breast_cancer_nn_wdbc.py
```

The script prints the dataset summary and the model summary, trains the network, and shows loss and accuracy curves (close the plot window to continue). It then prints the test results and example predictions.

## Notes

- Startup messages from TensorFlow (such as the oneDNN notice) are informational and can be ignored.
- Exact accuracy varies slightly between runs because of random weight initialization and floating-point differences.
- For a medical-style problem, accuracy alone is not enough. Check **recall for the malignant class**, since missing a malignant case is the costly error.

## Results

Fill these in after running the script:

| Metric | Value |
|--------|-------|
| Test accuracy | |
| Malignant recall | |
| Malignant precision | |

## Possible improvements

- Add `EarlyStopping` to stop training when validation loss stops improving
- Add `Dropout` or L2 regularization to reduce overfitting
- Tune the number of layers, units and learning rate
- Use cross-validation for a more reliable performance estimate

## Acknowledgements

Dataset created by Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian, University of Wisconsin. Obtained from the UCI Machine Learning Repository.

## Author

Md Atheeb ([GitHub](https://github.com/mdatheeb))
