# Load necessary libraries for data handling, visualization, and machine learning
from pandas import read_csv  # For reading CSV files into a DataFrame
from matplotlib import pyplot as plt  # For plotting graphs
from pandas.plotting import scatter_matrix  # For creating scatter plot matrices
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # For linear discriminant analysis (classification)
from sklearn.linear_model import LogisticRegression  # For logistic regression (classification)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # For data splitting and evaluation
from sklearn.naive_bayes import GaussianNB  # For Naive Bayes classification
from sklearn.neighbors import KNeighborsClassifier  # For K-Nearest Neighbors classification
from sklearn.svm import SVC  # For Support Vector Classification
from sklearn.tree import DecisionTreeClassifier  # For decision tree classification

# Define the path to the local CSV dataset
url = "bank_notes.csv"

# Specify column names for the dataset
names = ['Wavelet_Variance', 'Wavelet_Skewness', 'Wavelet_Kurtosis', 'Image_Entropy', 'Class']

# Load the dataset into a pandas DataFrame
dataset = read_csv(url, names=names)

# Display the number of rows and columns in the dataset
print("Shape of dataset:", dataset.shape)

# Display the first 20 rows to inspect the data
print("\nFirst 20 rows:")
print(dataset.head(20))

# Display a statistical summary (mean, std, min, max, quartiles) of each numeric column
print("\nStatistical summary:")
print(dataset.describe())

# Display the count of each class to understand class distribution
print("\nClass distribution:")
print(dataset.groupby('Class').size())

# Visualize distributions and outliers using box plots
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Visualize distributions using histograms
dataset.hist()
plt.show()

# Visualize pairwise relationships between features using a scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Prepare the data for model training and validation
array = dataset.values  # Convert DataFrame to NumPy array
X = array[:, 0:4]  # Select features (first 4 columns)
y = array[:, 4]  # Select target variable (Class column)

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Define a list of candidate machine learning models to evaluate
# Each tuple contains a short name and the model instance
models = [
    # Logistic Regression: predicts class probabilities using a linear combination of features
    ('LR', LogisticRegression(solver='lbfgs', max_iter=200)),

    # Linear Discriminant Analysis: finds linear combinations of features to separate classes
    ('LDA', LinearDiscriminantAnalysis()),

    # K-Nearest Neighbors: predicts class based on the majority class of nearest neighbors
    ('KNN', KNeighborsClassifier()),

    # Decision Tree (CART): splits the feature space into regions using a tree of decisions
    ('CART', DecisionTreeClassifier()),

    # Gaussian Naive Bayes: probabilistic classifier assuming feature independence and Gaussian distribution
    ('NB', GaussianNB()),

    # Support Vector Machine: finds an optimal hyperplane that separates classes in feature space
    ('SVM', SVC(gamma='auto'))
]

# Evaluate each model using 10-fold cross-validation
results = []  # To store cross-validation results
names = []  # To store model names
for name, model in models:
    # Define stratified K-fold to maintain class distribution in each fold
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    # Compute cross-validation accuracy scores for the current model
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    # Store the results
    results.append(cv_results)
    names.append(name)

    # Print mean accuracy and standard deviation for this model
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare model performance visually with a boxplot
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()

# Train the selected model (KNN in this case) on the full training dataset
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# Define new data points to make predictions on
X_values = [
    [3.6, 8.6, -2.3, -0.46],
    [0.7, -0.05, 5.7, -1.3],
    [-2.3, -0.7, 2.0, -0.4]
]

# Make predictions on the new data
predictions = model.predict(X_values)

# Print out predictions in a readable format
for i, pred in enumerate(predictions, start=1):
    # Convert numeric class to human-readable label
    outcome = "Fake" if pred == 0 else "Authentic"
    print(f"Bank note {i}: {outcome} (Class {pred})")