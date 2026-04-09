# Machine Learning Portfolio Projects

This repository contains **two Python machine learning projects** demonstrating data exploration, visualization, and classification. Both projects are fully reproducible and include their datasets, making them ideal for showcasing ML workflow, data analysis skills, and predictive modeling.

---

## Projects

### 1. Bank Note Authentication
- **Dataset:** `bank_notes.csv`  
- **Goal:** Predict whether a bank note is authentic or fake using wavelet features and image entropy.  
- **Features:** Wavelet Variance, Wavelet Skewness, Wavelet Kurtosis, Image Entropy  
- **Methods:** Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Naive Bayes, Support Vector Machines  
- **Highlights:**  
  - Data visualization: box plots, histograms, scatter matrix  
  - Algorithm comparison with cross-validation  
  - Prediction on new bank notes  

### 2. Student Stress Level Prediction
- **Dataset:** `student_stress_factors.csv`  
- **Goal:** Predict student stress levels based on sleep quality, headaches, performance, and study load.  
- **Features:** Sleep Quality, Headaches, Performance, Study Load  
- **Methods:** Decision Tree Classifier  
- **Highlights:**  
  - Data exploration: summary statistics, box plots, histograms, scatter matrix  
  - Fully reproducible predictions with fixed random state  
  - Predict stress levels for new student data  

---

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/jennabeachcodes/Machine-Learning
   ```   
2. Install required packages:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
3. Run the Python scripts for each project:
   ```bash
   python bank_notes_ml.py
   python student_stress_ml.py
   ```

## Key Skills Demonstrated
- Data loading and cleaning
- Univariate and bivariate analysis
- Data visualization (box plots, histograms, scatter matrices)
- Supervised machine learning (classification)
- Model evaluation and prediction
- Reproducible code for portfolio-quality projects

## License
This project is provided for educational purposes only as part of the course 
**26W-CST8400 - Analysis and Design Using Emerging Technologies** at Algonquin College, Ottawa, ON, Canada.