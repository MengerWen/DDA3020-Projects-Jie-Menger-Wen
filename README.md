# Project 1
## Task 1
- **3.2**: Code `gradient_descent` to optimize a 3x3 weight matrix $\mathbf{w}$ for softmax regression over 10,000 epochs (learning rate 0.1). Compute logits $\mathbf{z} = \mathbf{X} \mathbf{w}$, predictions with softmax, cross-entropy cost, and update $\mathbf{w}$ using the gradient. Output cost per epoch, decreasing from ~1.4265 to ~0.4085.
- **3.3**: Plot cost $J(\mathbf{w})$ vs. epochs, showing convergence from ~1.4 to ~0.4.
- **4.1**: Scatter plot `SepalLengthCm` vs. `SepalWidthCm` using ground-truth labels from `y` (one-hot, converted to 0, 1, 2), coloring classes (*setosa*, *versicolor*, *virginica*) as red, green, blue.
- **4.2**: Scatter plot as in 4.1, but using predicted labels from $\text{softmax}(\mathbf{X} \mathbf{w})$, with the same color scheme for comparison.

The solution must incorporate a bias term in $\mathbf{X}$, ensure softmax stability, and match provided cost and scatter plot outputs.
## Task 2
This question mandates the implementation of Support Vector Machines (SVMs) using `sklearn` for binary classification tasks, encompassing both linear and non-linear paradigms, as delineated in the `SVM_assignment1.ipynb` notebook. Specifically:
1. **Linear SVM**: Utilize the Iris dataset (Setosa vs. Versicolor) with petal length and width features, employing an `SVC` with a linear kernel and infinite regularization parameter (`C=float("inf")`) to enforce a hard margin. The task requires completing the `plot_svc_decision_boundary` function by deriving the decision boundary equation ($w_0 x_0 + w_1 x_1 + b = 0$, solved as $x_1 = -\frac{w_0 x_0 + b}{w_1}$) and margin offset ($\frac{1}{w_1}$) from the hyperplane coefficients and intercept, subsequently visualizing the boundary, margins, and support vectors on standardized data.
2. **Non-Linear SVM**: Apply a `Pipeline` to the `make_moons` dataset, integrating `PolynomialFeatures(degree=3)` for feature expansion into a higher-dimensional space, `StandardScaler` for feature normalization, and `LinearSVC(C=10, loss="hinge")` for soft-margin linear classification in the transformed space. The objective is to fit the model and visualize the resulting non-linear decision boundary.

The exercise demands proficiency in SVM optimization (maximizing margins), kernel trick approximations via polynomial features, and `sklearn` ’s preprocessing and visualization utilities, targeting a foundational grasp of SVM theory and application.
# Project 2
## Task 1
This question requires the implementation and comparative analysis of tree-based ensemble models on a subset of the Fashion-MNIST dataset for image classification. Specifically, it involves:
1. **Decision Tree Analysis:** Train a DecisionTreeClassifier with varying `max_depth` (1 to 9), compute training and testing accuracy using `accuracy_score`, and analyze overfitting trends via a plot of accuracy versus depth.
2. **Random Forest Analysis:** Train a RandomForestClassifier with the same `max_depth` range, evaluate training and testing accuracy, and compare its generalization and overfitting behavior against the Decision Tree through a similar accuracy-depth plot.
3. **Bagging vs. Random Forest Comparison:** Train BaggingClassifier and RandomForestClassifier with `n_estimators` values (10, 20, 50, 100), calculate test accuracy, and plot the results to assess the impact of ensemble size on performance, focusing on variance reduction and generalization.
4. **Hyperparameter Tuning:** Use GridSearchCV to perform 4-fold cross-validation on a RandomForestClassifier, tuning `n_estimators` (50, 100, 200) and `max_depth` (5, 10, 20, 50), and report the optimal parameters and corresponding test accuracy.

The tasks emphasize hyperparameter tuning, model evaluation, and comparative analysis of bias-variance trade-offs in tree-based models, leveraging scikit-learn for implementation and matplotlib for visualization.
## Task 2
This question mandates the implementation and evaluation of two Convolutional Neural Network (CNN) models using the TensorFlow framework on the Fashion-MNIST dataset. The objective is to investigate the efficacy of CNNs for image classification tasks, contrasting their performance with traditional machine learning approaches (e.g., tree-based models from Question 3). The assignment comprises two distinct tasks:
1. Design and implement a rudimentary CNN model featuring:
	- A single 2D convolutional layer with 16 filters (3x3 kernel, ReLU activation).
	- A 2D max-pooling layer (2x2 window).
	- A flattening layer to transform 2D feature maps into a 1D vector.
	- A fully connected layer with Softmax activation for 10-class classification.
	- The model must employ categorical cross-entropy as the loss function, optimized via the Adam algorithm, trained over 2 epochs, and evaluated for classification accuracy on the test set.

2. Develop a more sophisticated CNN model to outperform Task 1 in terms of test set accuracy. A reference architecture is provided, segmented into three components:
	- **Primary Feature Extraction**: A 2D convolutional layer with 32 filters (3x3, ReLU), followed by batch normalization, a 2x2 max-pooling layer, and a dropout layer (25% rate) for regularization.
	- **Advanced Feature Extraction**: A similar structure with 64 filters (3x3, ReLU), batch normalization, 2x2 max-pooling, and 25% dropout to capture higher-level features.
	- **Classification**: A flattening layer, a dense layer with 512 units (ReLU), and a final dense layer with Softmax activation for classification.
	- The model retains categorical cross-entropy loss and Adam optimization, with training over 2 epochs and subsequent test set evaluation.

**Technical Specifications**:
- **Data Preprocessing**: The Fashion-MNIST dataset (28x28 grayscale images) is normalized to [0, 1] and reshaped to (samples, 28, 28, 1), with labels one-hot encoded for 10 classes.
- **Evaluation Metric**: Performance is quantified via test set accuracy, with Task 2 required to surpass Task 1.
- **Environment**: TensorFlow 2.9 is specified, necessitating precise configuration of convolutional, pooling, normalization, and regularization layers.

**Purpose**: By juxtaposing a simple and a complex CNN, the task elucidates the hierarchical feature extraction capabilities of deep learning and the impact of regularization techniques (e.g., Dropout, BatchNormalization) on mitigating overfitting in data-intensive applications.
