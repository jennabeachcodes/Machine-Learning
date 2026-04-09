# Load libraries
from pandas import read_csv                     # For reading CSV files into a DataFrame
from matplotlib import pyplot as plt            # For plotting graphs
from pandas.plotting import scatter_matrix      # For scatter plot matrices
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
url = "student_stress_factors.csv"             # Path to dataset file
names = ['Sleep_Quality', 'Headaches', 'Performance', 'Study_Load', 'Stress_Level']  # Column names
dataset = read_csv(url, names=names)

# Display dataset shape and first 10 rows
print("Shape of dataset:", dataset.shape)
print("\nFirst 10 rows:")
print(dataset.head(10))

# Univariate analysis
print("\nStatistical Summary:")
print(dataset.describe())                       # Summary stats for numerical features
print("\nStress Level Distribution:")
print(dataset.groupby('Stress_Level').size())  # Counts per stress level

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)  # Box plots
plt.suptitle("Box Plots of Features")
plt.show()

dataset.hist()                                 # Histograms
plt.suptitle("Histograms of Features")
plt.show()

# Bivariate analysis
scatter_matrix(dataset)                        # Scatter plot matrix
plt.suptitle("Scatter Plot Matrix")
plt.show()

# Prepare data for modeling
array = dataset.values
X = array[:,0:4]                               # Features: Sleep, Headaches, Performance, Study Load
y = array[:,4]                                 # Target: Stress_Level

# Split dataset into training (80%) and validation (20%) with fixed random_state
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1       # Fix random_state for reproducibility
)

# Train Decision Tree model with fixed random_state
model = DecisionTreeClassifier(random_state=1)  # Ensures tree splits are reproducible
model.fit(X_train, Y_train)

# Predict stress levels for new students
X_new = [[1, 1, 1, 4],
         [3, 1, 2, 5],
         [1, 2, 3, 4]]

predictions = model.predict(X_new)

# Print predictions
for i, pred in enumerate(predictions, start=1):
    print(f"Student {i}: Predicted Stress Level = {pred}")