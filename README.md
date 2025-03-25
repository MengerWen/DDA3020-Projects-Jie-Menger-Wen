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

The exercise demands proficiency in SVM optimization (maximizing margins), kernel trick approximations via polynomial features, and `sklearn`’s preprocessing and visualization utilities, targeting a foundational grasp of SVM theory and application.