from models.observation import PipelineStage, BugSymptom

TASKS = {
    "task_1": {
        "pipeline_context": (
            "A binary classification pipeline on tabular data. "
            "The model is a simple logistic regression trained on a CSV dataset. "
            "Validation accuracy is stuck at ~0.5 despite clean data."
        ),
        "bugs": [
            {
                "stage": PipelineStage.PREPROCESSING,
                "symptom": (
                    "Features have vastly different scales. The model is not converging properly. "
                    "HINT: The scaler is being fitted on the test set — this causes data leakage."
                ),
                "code_snippet": (
                    "scaler = MinMaxScaler(feature_range=(0, 1))\n"
                    "X_train = scaler.fit_transform(X_train)\n"
                    "X_test = scaler.fit_transform(X_test)  # BUG: fit_transform on test set"
                ),
                "correct_keywords": [
                    "transform", "scaler.transform(X_test)", "fit only on train",
                    "only fit on train", "fit_transform only on", "data leakage",
                    "transform(x_test)", "transform only"
                ],
                "partial_keywords": ["scaler", "test set", "leakage", "fit_transform"],
            }
        ],
    },

    "task_2": {
        "pipeline_context": (
            "A multi-class image classification pipeline using a CNN. "
            "Training loss is NaN after the first epoch. "
            "The dataset has 5 classes."
        ),
        "bugs": [
            {
                "stage": PipelineStage.PREPROCESSING,
                "symptom": (
                    "Images are loaded as uint8 (0-255) and passed directly to the model without normalization. "
                    "HINT: Pixel values must be scaled to [0.0, 1.0] by dividing by 255."
                ),
                "code_snippet": (
                    "images = np.array([cv2.imread(p) for p in paths])  # dtype: uint8\n"
                    "# No normalization applied\n"
                    "dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))"
                ),
                "correct_keywords": [
                    "/ 255", "normalize", "float32", "0.0 to 1.0", "divide by 255",
                    "/255", "astype(float", "to(torch.float", ".float()", "pixel"
                ],
                "partial_keywords": ["dtype", "uint8", "scale", "image", "normalization"],
            },
            {
                "stage": PipelineStage.MODEL_CONFIG,
                "symptom": (
                    "Loss becomes NaN after first batch. BCELoss is being used but there are 5 classes. "
                    "HINT: BCELoss is for binary tasks only. Use CrossEntropyLoss for multi-class."
                ),
                "code_snippet": (
                    "criterion = nn.BCELoss()  # BUG: BCELoss used for multi-class\n"
                    "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n"
                    "# model final layer: nn.Linear(128, 5)"
                ),
                "correct_keywords": [
                    "CrossEntropyLoss", "NLLLoss", "cross entropy", "cross_entropy",
                    "nn.crossentropyloss", "replace bce", "replace bceloss",
                    "multi-class loss", "categorical crossentropy"
                ],
                "partial_keywords": ["loss", "BCE", "binary", "wrong loss", "multi-class", "multiclass", "5 class"],
            },
        ],
    },

    "task_3": {
        "pipeline_context": (
            "A regression pipeline predicting house prices. "
            "Model is an MLP with 3 hidden layers. "
            "Training loss decreases but validation loss explodes after epoch 3. "
            "There are exactly 3 bugs — one in each stage. Fix them IN ORDER: "
            "first preprocessing, then model_config, then training_loop."
        ),
        "bugs": [
            {
                "stage": PipelineStage.PREPROCESSING,
                "symptom": (
                    "Missing values in 3 columns are being dropped entirely, losing 40% of training data. "
                    "HINT: Use imputation (fillna, SimpleImputer) instead of dropna."
                ),
                "code_snippet": (
                    "df = df.dropna()  # BUG: drops rows with any NaN, losing 40% of data\n"
                    "X = df.drop('price', axis=1)\n"
                    "y = df['price']"
                ),
                "correct_keywords": [
                    "fillna", "impute", "SimpleImputer", "median", "mean imputation",
                    "mean()", "replace nan", "fill missing", "knnimputer"
                ],
                "partial_keywords": ["missing", "NaN", "dropna", "imputation", "drop", "nan"],
            },
            {
                "stage": PipelineStage.MODEL_CONFIG,
                "symptom": (
                    "Validation loss explodes after epoch 3. Learning rate is set to 0.1 which is too high. "
                    "HINT: Reduce learning rate to 0.001 or 0.0001."
                ),
                "code_snippet": (
                    "optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # BUG: lr too high\n"
                    "# No regularization, no dropout\n"
                    "scheduler = None"
                ),
                "correct_keywords": [
                    "lr=0.001", "lr=0.0001", "lower learning rate", "reduce lr",
                    "1e-3", "1e-4", "0.001", "0.0001", "decrease lr", "smaller lr"
                ],
                "partial_keywords": ["learning rate", "lr", "too high", "optimizer", "explod"],
            },
            {
                "stage": PipelineStage.TRAINING_LOOP,
                "symptom": (
                    "Gradients accumulate across batches. Loss is unstable and grows over time. "
                    "HINT: Call optimizer.zero_grad() before loss.backward() in each batch."
                ),
                "code_snippet": (
                    "for batch in dataloader:\n"
                    "    inputs, targets = batch\n"
                    "    outputs = model(inputs)\n"
                    "    loss = criterion(outputs, targets)\n"
                    "    loss.backward()\n"
                    "    optimizer.step()  # BUG: optimizer.zero_grad() never called"
                ),
                "correct_keywords": [
                    "zero_grad", "optimizer.zero_grad()", "clear gradients",
                    "zero grad", "zeroing", ".zero_grad", "zero the grad",
                    "reset gradient", "reset grad", "gradient accumulation",
                    "add optimizer.zero_grad", "call zero_grad", "missing zero_grad"
                ],
                "partial_keywords": [
                    "gradient", "accumulate", "zero", "grad", "backward",
                    "before loss", "each batch", "each iteration", "training loop"
                ],
            },
        ],
    },
}