from models.observation import PipelineStage, BugSymptom

# Each task is a list of stages with bugs.
# Each stage has:
#   - stage: PipelineStage
#   - symptom: what the agent observes going wrong
#   - code_snippet: the buggy code shown to the agent
#   - correct_keywords: list of strings; agent fix must contain at least one to be "correct"
#   - partial_keywords: list of strings; if fix contains these but not correct_keywords → partial
#   - pipeline_context: description of the overall pipeline

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
                "symptom": "Features have vastly different scales. The model is not converging properly.",
                "code_snippet": (
                    "scaler = MinMaxScaler(feature_range=(0, 1))\n"
                    "X_train = scaler.fit_transform(X_train)\n"
                    "X_test = scaler.fit_transform(X_test)  # BUG: fit_transform on test set"
                ),
                "correct_keywords": ["transform", "fit only on train", "scaler.transform(X_test)"],
                "partial_keywords": ["scaler", "test set", "leakage"],
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
                "symptom": "Images are loaded as uint8 (0-255) and passed directly to the model without normalization.",
                "code_snippet": (
                    "images = np.array([cv2.imread(p) for p in paths])  # dtype: uint8\n"
                    "# No normalization applied\n"
                    "dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))"
                ),
                "correct_keywords": ["/ 255", "normalize", "float32", "0.0 to 1.0", "divide by 255"],
                "partial_keywords": ["dtype", "uint8", "scale"],
            },
            {
                "stage": PipelineStage.MODEL_CONFIG,
                "symptom": "Loss becomes NaN after first batch. Model output shape looks wrong.",
                "code_snippet": (
                    "criterion = nn.BCELoss()  # BUG: BCELoss used for multi-class\n"
                    "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n"
                    "# model final layer: nn.Linear(128, 5)"
                ),
                "correct_keywords": ["CrossEntropyLoss", "NLLLoss", "multi-class loss"],
                "partial_keywords": ["loss", "BCE", "binary", "wrong loss"],
            },
        ],
    },

    "task_3": {
        "pipeline_context": (
            "A regression pipeline predicting house prices. "
            "Model is an MLP with 3 hidden layers. "
            "Training loss decreases but validation loss explodes after epoch 3. "
            "Suspected issues span all pipeline stages."
        ),
        "bugs": [
            {
                "stage": PipelineStage.PREPROCESSING,
                "symptom": "Missing values in 3 columns are being dropped entirely, losing 40% of training data.",
                "code_snippet": (
                    "df = df.dropna()  # BUG: drops rows with any NaN, losing 40% of data\n"
                    "X = df.drop('price', axis=1)\n"
                    "y = df['price']"
                ),
                "correct_keywords": ["fillna", "impute", "SimpleImputer", "median", "mean imputation"],
                "partial_keywords": ["missing", "NaN", "dropna", "imputation"],
            },
            {
                "stage": PipelineStage.MODEL_CONFIG,
                "symptom": "Validation loss explodes after epoch 3. Model may be overfitting immediately.",
                "code_snippet": (
                    "optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # BUG: lr too high\n"
                    "# No regularization, no dropout\n"
                    "scheduler = None"
                ),
                "correct_keywords": ["lr=0.001", "lr=0.0001", "lower learning rate", "reduce lr", "1e-3", "1e-4"],
                "partial_keywords": ["learning rate", "lr", "too high", "optimizer"],
            },
            {
                "stage": PipelineStage.TRAINING_LOOP,
                "symptom": "Gradients accumulate across batches. Loss is unstable and grows over time.",
                "code_snippet": (
                    "for batch in dataloader:\n"
                    "    inputs, targets = batch\n"
                    "    outputs = model(inputs)\n"
                    "    loss = criterion(outputs, targets)\n"
                    "    loss.backward()\n"
                    "    optimizer.step()  # BUG: optimizer.zero_grad() never called"
                ),
                "correct_keywords": ["zero_grad", "optimizer.zero_grad()", "clear gradients"],
                "partial_keywords": ["gradient", "accumulate", "zero grad"],
            },
        ],
    },
}