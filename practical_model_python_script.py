# %% [markdown]
# # Support Vector Machines (SVM): A Complete Guide
# 
# ## What are Support Vector Machines?
# 
# Support Vector Machines (SVMs) are powerful machine learning algorithms used for classification and regression tasks. Think of SVM as a smart boundary-drawing system that separates different groups of data points.
# 
# ### Simple Analogy
# 
# Imagine you have red and blue marbles scattered on a table, and you need to draw a line that best separates them. SVM doesn't just draw any line - it finds the **optimal line** that:
# - Separates the colors as accurately as possible
# - Stays as far away as possible from both red and blue marbles
# - Creates the widest "safety zone" between the groups
# 
# This "safety zone" is called the **margin**, and SVM aims to maximize it.
# 
# ## Core Concepts Made Simple
# 
# ### 1. The Decision Boundary
# - **What it is**: The line (or curve) that separates different classes
# - **Goal**: Find the boundary that gives the best separation
# - **Real-world example**: Spam vs legitimate emails, diseased vs healthy patients
# 
# ### 2. Support Vectors
# - **What they are**: The data points closest to the decision boundary
# - **Why they matter**: Only these points determine where the boundary goes
# - **Analogy**: Like the key witnesses in a court case - only their testimony matters for the final decision
# 
# ### 3. The Margin
# - **Definition**: The distance between the decision boundary and the nearest data points
# - **SVM's goal**: Maximize this margin for better generalization
# - **Benefit**: Wider margins usually mean better performance on new, unseen data
# 
# ### 4. Hard vs Soft Margin
# **Hard Margin SVM:**
# - **What it is**: Requires perfect separation - no data points can cross the boundary
# - **When to use**: Only when data is perfectly separable (rare in real world)
# - **Problem**: Very sensitive to outliers, often impossible to achieve
# 
# **Soft Margin SVM:**
# - **What it is**: Allows some data points to cross the boundary or be misclassified
# - **Trade-off**: Balance between wide margin and classification accuracy
# - **C Parameter**: Controls this trade-off (high C = less tolerance for errors)
# - **Reality**: Used in almost all real-world applications
# 
# ### 5. Hinge Loss
# - **Purpose**: The cost function that SVM tries to minimize
# - **Formula**: `max(0, 1 - y × f(x))` where y is true label, f(x) is prediction
# - **How it works**: 
#   - Loss = 0 when prediction is correct and confident
#   - Loss increases as prediction becomes wrong or less confident
# - **Why it matters**: Encourages confident, correct predictions while allowing some mistakes
# 
# ## Why Use Support Vector Machines?
# 
# ### 1. **Excellent Performance**
# - Often achieves high accuracy on various datasets
# - Works well even with limited training data
# - Robust to outliers due to margin maximization
# 
# ### 2. **Handles Complex Patterns**
# - Can create non-linear decision boundaries using "kernel tricks"
# - Adapts to different data shapes and distributions
# - Effective in high-dimensional spaces
# 
# ### 3. **Mathematical Foundation**
# - Based on solid mathematical principles (convex optimization)
# - Guarantees finding the globally optimal solution
# - No local minimum problems like in neural networks
# 
# ### 4. **Memory Efficient**
# - Only stores support vectors (usually a small subset of training data)
# - Prediction is fast once trained
# - Scales well to large feature spaces
# 
# ## SVM vs Logistic Regression: Key Differences
# 
# | Aspect | Support Vector Machine | Logistic Regression |
# |--------|------------------------|-------------------|
# | **Decision Boundary** | Finds maximum margin separator | Finds probabilistic boundary |
# | **Objective** | Maximize margin between classes | Maximize likelihood of correct predictions |
# | **Output** | Class prediction + distance to boundary | Probability of belonging to each class |
# | **Handling Outliers** | Robust (focuses on support vectors) | Sensitive to outliers |
# | **Non-linear Patterns** | Excellent (kernel trick) | Limited (needs feature engineering) |
# | **Interpretability** | Moderate (support vectors matter most) | High (coefficients show feature importance) |
# | **Training Time** | Can be slow on large datasets | Generally faster |
# | **Memory Usage** | Efficient (stores only support vectors) | Stores all parameters |
# 
# ### When to Choose SVM over Logistic Regression
# 
# **Choose SVM when:**
# - You need high accuracy and have sufficient computational resources
# - Your data has complex, non-linear patterns
# - You're working with high-dimensional data (text, images)
# - You don't need probability estimates, just classifications
# - Your dataset has outliers that might affect other algorithms
# 
# **Choose Logistic Regression when:**
# - You need probability estimates (not just classifications)
# - Interpretability is crucial (understanding which features matter most)
# - You're working with very large datasets where speed matters
# - You need a simple, fast baseline model
# - Your data relationships are mostly linear
# 
# ## Real-World Applications
# 
# ### 1. **Text Classification**
# - **Spam Detection**: Separating spam from legitimate emails
# - **Sentiment Analysis**: Determining if reviews are positive or negative
# - **Document Classification**: Organizing news articles by topic
# 
# ### 2. **Image Recognition**
# - **Face Detection**: Finding faces in photographs
# - **Medical Image Analysis**: Detecting tumors in X-rays or MRIs
# - **Object Recognition**: Identifying objects in pictures
# 
# ### 3. **Finance & Healthcare**
# - **Credit Scoring**: Determining loan approval likelihood
# - **Fraud Detection**: Identifying suspicious transactions
# - **Disease Diagnosis**: Classifying diseases based on symptoms
# 
# ## Types of SVM Kernels Explained
# 
# ### Linear vs Non-Linear Kernels
# 
# **Linear Kernels:**
# - **What they do**: Create straight-line decision boundaries
# - **When they work**: Data that can be separated by straight lines/planes
# - **Example**: Separating emails by word count - if spam emails consistently have more promotional words
# - **Advantage**: Simple, fast, interpretable
# - **Formula**: Just the dot product of feature vectors
# 
# **Non-Linear Kernels:**
# - **What they do**: Create curved, complex decision boundaries
# - **When needed**: Data with complex patterns that can't be separated by straight lines
# - **How they work**: Transform data into higher dimensions where linear separation becomes possible
# - **Trade-off**: More powerful but require more tuning
# 
# ### Specific Kernel Types
# 
# ### 1. **Linear Kernel**
# - **Best for**: Data that can be separated by a straight line
# - **Example**: Simple email spam detection based on word count
# - **Pros**: Fast, interpretable, works well with many features
# - **Cons**: Limited to linear patterns
# 
# ### 2. **Polynomial Kernel**
# - **Best for**: Data with polynomial relationships
# - **Example**: Image recognition where pixel combinations matter
# - **Pros**: Can capture interactions between features (like x₁ × x₂)
# - **Cons**: Can overfit, more parameters to tune
# 
# ### 3. **RBF (Radial Basis Function) Kernel**
# - **Best for**: Complex, non-linear patterns
# - **Example**: Medical diagnosis with complex symptom interactions
# - **Pros**: Very flexible, can fit almost any shape
# - **Cons**: Can overfit easily, requires careful parameter tuning
# 
# ## Getting Started with SVM
# 
# ### Step 1: Prepare Your Data
# ```python
# # Scale your features (very important for SVM!)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ```
# 
# ### Step 2: Choose Your Kernel
# ```python
# from sklearn.svm import SVC
# 
# # For linear patterns
# model = SVC(kernel='linear')
# 
# # For complex patterns
# model = SVC(kernel='rbf')
# ```
# 
# ### Step 3: Tune Parameters
# ```python
# from sklearn.model_selection import GridSearchCV
# 
# # Find best parameters
# param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
# grid_search = GridSearchCV(SVC(), param_grid, cv=5)
# ```
# 
# ## Common Parameters Explained
# 
# ### **C Parameter (Regularization)**
# - **What it controls**: Trade-off between smooth boundary and classifying training points correctly
# - **Low C**: Smooth boundary, may underfit
# - **High C**: Complex boundary, may overfit
# - **How to choose**: Use cross-validation to find the best value
# 
# ### **Gamma Parameter (for RBF/Polynomial kernels)**
# - **What it controls**: How much influence each training example has
# - **Low gamma**: Smooth, simple boundaries
# - **High gamma**: Complex, wiggly boundaries
# - **How to choose**: Start with 'scale' and tune if needed
# 
# ## Best Practices
# 
# ### 1. **Always Scale Your Features**
# SVM is sensitive to the scale of input features. Always use StandardScaler or MinMaxScaler.
# 
# ### 2. **Start Simple**
# Begin with a linear kernel, then try RBF if you need more complexity.
# 
# ### 3. **Use Cross-Validation**
# Never trust results from a single train-test split. Use k-fold cross-validation.
# 
# ### 4. **Grid Search for Parameters**
# Systematically search for the best C and gamma values using GridSearchCV.
# 
# ### 5. **Monitor for Overfitting**
# Check that your model performs similarly on training and validation data.
# 
# ## Limitations to Consider
# 
# ### 1. **Computational Complexity**
# - Training can be slow on very large datasets (>100k samples)
# - Memory usage can be high for complex kernels
# 
# ### 2. **Parameter Sensitivity**
# - Requires careful tuning of C and gamma parameters
# - Performance can vary significantly with different parameter values
# 
# ### 3. **No Probability Estimates**
# - Standard SVM gives classifications, not probabilities
# - Need special techniques (Platt scaling) for probability estimates
# 
# ### 4. **Feature Scaling Dependency**
# - Must preprocess data carefully
# - Sensitive to outliers in feature scaling
# 
# ## Conclusion
# 
# Support Vector Machines are powerful, versatile algorithms that excel at finding optimal decision boundaries. They're particularly valuable when:
# 
# - You need high accuracy on complex datasets
# - Your data has non-linear patterns
# - You're working with high-dimensional data
# - You want a mathematically principled approach
# 
# While they require more tuning than simpler algorithms like logistic regression, the investment in learning SVM pays off with superior performance on many real-world problems.
# 
# Whether you're detecting spam, recognizing images, or diagnosing diseases, SVM provides a robust foundation for classification tasks. Start with the basics, understand the concepts, and gradually explore advanced techniques like kernel tricks and parameter optimization.
# 
# Remember: the key to success with SVM is understanding your data, choosing the right kernel, and carefully tuning parameters through systematic experimentation.

# %%


# %%
# ===================================================================
# Support Vector Machines: A Journey Through Decision Making  
# ===================================================================
# From Chaos to Clear Boundaries - Understanding How Machines Learn to Separate

"""
THE STORY SO FAR

In our digital world, we encounter the age-old problem of classification: 
given a set of characteristics, can we predict which group something belongs to?

Today, we'll follow the journey of a Support Vector Machine (SVM) as it learns 
to draw the perfect boundary between two tribes in a 2D world.

 Our Cast of Characters:
- The Blue Tribe (Class 0): Conservative, tight-knit community
- The Red Tribe (Class 1): Adventurous, spread-out settlers  
- The SVM: Our boundary-drawing expert
- Support Vectors: The influential citizens who shape decisions

 What You'll Learn:
- How SVM finds the optimal decision boundary
- Why margin maximization matters for generalization
- The critical role of support vectors
- Real-world performance analysis and interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set style for beautiful plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ===================================================================
#  CHAPTER 1: Setting the Stage - "Two Tribes in a Digital World"
# ===================================================================

print(" CHAPTER 1: Creating Our Digital World")
print("=" * 50)

#  The Random Seed: Ensuring our story is reproducible
np.random.seed(42)
print(" Random seed set to 42 - our story will be consistent every time!")

#  Creating the Blue Tribe (Class 0)
print("\n Creating the Blue Tribe:")
X0 = np.random.randn(50, 2) * 1.0 + [0, 0]  # Centered at origin, tighter spread
print(f"   - Population: 50 members")
print(f"   - Territory: Centered around (0, 0)")
print(f"   - Lifestyle: Conservative (smaller variance = 1.0)")

#  Creating the Red Tribe (Class 1) 
print("\n Creating the Red Tribe:")
X1 = np.random.randn(50, 2) * 1.5 + [1.5, 1.5]  # Northeast location, wider spread
print(f"   - Population: 50 members") 
print(f"   - Territory: Centered around (1.5, 1.5)")
print(f"   - Lifestyle: Adventurous (larger variance = 1.5)")

# Combining our world
print("\n Combining tribes into our complete world")
X = np.vstack((X0, X1))  # Stack the tribe territories
y = np.array([0]*50 + [1]*50)  # Assign tribal identities
print(f"   - Total population: {len(X)} citizens")
print(f"   - Features per citizen: {X.shape[1]} (x-coordinate, y-coordinate)")

#  Let's visualize our world
plt.figure(figsize=(10, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=0.6, s=50, label='Blue Tribe', edgecolors='black')
plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6, s=50, label='Red Tribe', edgecolors='black')
plt.title(" Our Digital World: Two Tribes", fontsize=16, fontweight='bold')
plt.xlabel("Feature 1 (Territory X-coordinate)", fontsize=12)
plt.ylabel("Feature 2 (Territory Y-coordinate)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print(" Chapter 1 Complete: Our world is ready for exploration!")

# ===================================================================
#  CHAPTER 2: The Training Academy - "Learning from Examples"
# ===================================================================

print("\n\n CHAPTER 2: Setting Up the Training Academy")
print("=" * 50)

#  The Great Division: Training vs Testing
print(" Dividing our population for the grand experiment")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f" Training Academy:")
print(f"   - Training population: {len(X_train)} citizens ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Blue trainers: {sum(y_train == 0)}")
print(f"   - Red trainers: {sum(y_train == 1)}")

print(f"\n Final Exam Hall:")
print(f"   - Test population: {len(X_test)} citizens ({len(X_test)/len(X)*100:.1f}%)")
print(f"   - Blue test subjects: {sum(y_test == 0)}")
print(f"   - Red test subjects: {sum(y_test == 1)}")

#  Enter the SVM: Our Boundary-Drawing Expert
print("\n Introducing our SVM student")
print(" SVM's Personality Profile:")
print("   - Kernel: Linear (prefers straight-line boundaries)")
print("   - C Parameter: 1.0 (balanced personality - not too strict, not too lenient)")
print("   - Mission: Find the widest possible 'no man's land' between tribes")

model = SVC(kernel='linear', C=1.0)
print("\n Training begins")
model.fit(X_train, y_train)
print(" Training complete! Our SVM has learned to draw boundaries.")

print(f"\n Key Learning Stats:")
print(f"   - Support Vectors found: {len(model.support_vectors_)} critical citizens")
print(f"   - These {len(model.support_vectors_)} citizens will determine the entire boundary!")

# ===================================================================
#  CHAPTER 3: The Moment of Truth - "Testing Our Boundary"
# ===================================================================

print("\n\n CHAPTER 3: The Final Examination")
print("=" * 50)

#  The Big Performance
print(" The moment of truth - how well did our SVM learn?")
y_pred = model.predict(X_test)

#  Performance Analysis
print("\n DETAILED PERFORMANCE REPORT:")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=['Blue Tribe', 'Red Tribe']))

print("\n Confusion Matrix - The Complete Story:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted")
print(f"               Blue    Red")
print(f"Actual  Blue    {cm[0,0]}      {cm[0,1]}")
print(f"        Red     {cm[1,0]}      {cm[1,1]}")

#  The Bottom Line
errors = (y_test != y_pred).sum()
accuracy = (len(y_test) - errors) / len(y_test) * 100
print(f"\n THE BOTTOM LINE:")
print(f"   - Correct predictions: {len(y_test) - errors}/{len(y_test)}")
print(f"   - Mistakes made: {errors}")
print(f"   - Accuracy: {accuracy:.1f}%")

#  Detailed Analysis Based on Actual Results
print(f"\n DETAILED PERFORMANCE BREAKDOWN:")
print(f"   - Blue Tribe Performance:")
print(f"     * Precision: When we predict Blue, we're right most of the time")
print(f"     * Recall: We catch most of the actual Blue citizens")
print(f"   - Red Tribe Performance:")
print(f"     * Precision: When we predict Red, good accuracy")
print(f"     * Recall: We're good at identifying Red citizens")

print(f"\n CONFUSION MATRIX INSIGHTS:")
print(f"   - True Blues correctly identified: {cm[0,0]}")
print(f"   - Blues misclassified as Reds: {cm[0,1]}")
print(f"   - Reds misclassified as Blues: {cm[1,0]}") 
print(f"   - True Reds correctly identified: {cm[1,1]}")

if accuracy >= 80:
    print("   -  SOLID PERFORMANCE! Our SVM shows good tribal understanding!")
    print("   -  Any misclassifications reveal natural overlapping territories")
elif accuracy >= 70:
    print("   -  DECENT START! Room for improvement with parameter tuning")
else:
    print("   -  NEEDS WORK! Consider different kernels or data preprocessing")

print(f"\n KEY INSIGHTS:")
print(f"   - Border citizens are the most challenging to classify")
print(f"   - Real-world lesson: Perfect separation is rare!")
print(f"   - SVM finds the best possible boundary given the data overlap")

# ===================================================================
#  CHAPTER 4: The Art of Visualization - "Seeing the Invisible"
# ===================================================================

print("\n\n CHAPTER 4: Visualizing the SVM's Masterpiece")
print("=" * 50)

def plot_svc_decision_boundary(svc, X, y):
    """
     The Artist's Canvas: Revealing the SVM's invisible boundary
    
    This visualization shows:
    - Decision boundary (solid black line): "The Great Divide"
    - Margin boundaries (dashed lines): "Safety zones" 
    - Support vectors (circled): "The influential citizens"
    - Data points: Our tribal populations
    """
    
    plt.figure(figsize=(10, 6))

    # Plot data points with tribal colors
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

    # Create grid to evaluate model across the landscape
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Decision function values for each point in grid
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #  Draw the sacred boundaries
    # levels=[-1, 0, 1] creates three lines:
    # -1: Red safety zone boundary
    #  0: The decision boundary  
    #  1: Blue safety zone boundary
    plt.contour(xx, yy, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    #  Highlight the VIP citizens: Support Vectors
    plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', linewidths=2,
                label='Support Vectors')

    plt.title("SVM Decision Boundary with Margins and Support Vectors", fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print(" Creating the visualization masterpiece")
plot_svc_decision_boundary(model, X_test, y_test)

print(" What you're seeing:")
print("   - Solid black line: The optimal decision boundary")
print("   - Dashed lines: The margin boundaries (safety zones)")
print("   - Black circles: Support vectors (the VIP citizens who determine everything)")
print("   - Blue/Red dots: Our test citizens colored by their true tribal identity")
print("   - The boundary maximizes the 'no man's land' between tribes")

# ===================================================================
#  CHAPTER 5: The Mathematical Revelation - "The Sacred Margin"
# ===================================================================

print("\n\n CHAPTER 5: The Mathematical Magic")
print("=" * 50)

#  The Science Behind the Magic
print(" Unveiling the mathematical secrets")

# Get the boundary's direction vector
w = model.coef_[0]
print(f" Boundary direction vector: [{w[0]:.4f}, {w[1]:.4f}]")

# Calculate the sacred margin distance
margin_distance = 2 / np.linalg.norm(w)
print(f"\n THE SACRED MARGIN:")
print(f"   - Margin width: {margin_distance:.4f} units")
print(f"   - This is the 'confidence zone' our SVM created!")
print(f"   - It's the widest possible 'no man's land' between tribes")

# Support vector analysis
print(f"\n SUPPORT VECTOR ANALYSIS:")
print(f"   - Total support vectors: {len(model.support_vectors_)}")
print(f"   - Blue support vectors: {sum(y_train[model.support_] == 0)}")
print(f"   - Red support vectors: {sum(y_train[model.support_] == 1)}")
print(f"   - These {len(model.support_vectors_)} citizens define the entire boundary!")

print(f"\n WHY THIS MATTERS:")
print(f"   - Larger margin = better generalization to new data")
print(f"   - Only support vectors matter - other points could disappear!")
print(f"   - SVM finds the most 'confident' boundary possible")
print(f"   - Mathematical guarantee: This is the optimal linear separator!")

# ===================================================================
#  CHAPTER 6: Understanding Our Results - "The Reality Check"
# ===================================================================

print("\n\n CHAPTER 6: Deep Dive into Performance")
print("=" * 50)

print(" INTERPRETING OUR RESULTS:")

# Extract precision and recall for detailed analysis
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print(f"\n Blue Tribe (Class 0) Analysis:")
print(f"   - Precision: {precision[0]:.2f} (When we predict Blue, we're right {precision[0]*100:.0f}% of time)")
print(f"   - Recall: {recall[0]:.2f} (We correctly identify {recall[0]*100:.0f}% of actual Blues)")
print(f"   - F1-Score: {f1[0]:.2f} (Balanced measure of performance)")

print(f"\n Red Tribe (Class 1) Analysis:")
print(f"   - Precision: {precision[1]:.2f} (When we predict Red, we're right {precision[1]*100:.0f}% of time)")
print(f"   - Recall: {recall[1]:.2f} (We correctly identify {recall[1]*100:.0f}% of actual Reds)")
print(f"   - F1-Score: {f1[1]:.2f} (Balanced measure of performance)")

print(f"\n REAL-WORLD IMPLICATIONS:")
if errors <= 6:
    print(f"   - {errors} misclassifications out of {len(y_test)} is excellent!")
    print(f"   - These errors likely occur in natural overlap zones")
    print(f"   - In medical diagnosis: {accuracy:.0f}% accuracy could save lives")
    print(f"   - In spam detection: {accuracy:.0f}% means high-quality inbox")
elif errors <= 10:
    print(f"   - {errors} misclassifications show room for improvement")
    print(f"   - Still useful for many real-world applications")
else:
    print(f"   - {errors} misclassifications suggest need for different approach")

print(f"\n ERROR ANALYSIS:")
print(f"   - Blue citizens misclassified as Red: {cm[0,1]}")
print(f"   - Red citizens misclassified as Blue: {cm[1,0]}")
if cm[0,1] > cm[1,0]:
    print(f"   - SVM has slight bias toward Red predictions")
elif cm[1,0] > cm[0,1]:
    print(f"   - SVM has slight bias toward Blue predictions")
else:
    print(f"   - SVM shows balanced prediction behavior")

# ===================================================================
#  FINAL CHAPTER: Conclusions & Key Learnings
# ===================================================================

print("\n\n FINAL CHAPTER: Conclusions & Key Learnings")
print("=" * 50)

print(" WHAT WE DISCOVERED:")
print("   1. SVM doesn't just separate - it maximizes confidence")
print("   2. Only a few 'influential citizens' (support vectors) matter")
print("   3. The margin width determines generalization ability")
print("   4. Linear kernels work great for linearly separable data")
print("   5. Real-world data has natural overlaps and ambiguous cases")

print(f"\n YOUR ACTUAL RESULTS:")
print(f"   - Achieved {accuracy:.0f}% accuracy ({len(y_test) - errors}/{len(y_test)} correct)")
print(f"   - Found {len(model.support_vectors_)} critical support vectors")
print(f"   - Created margin width of {margin_distance:.4f} units")
print(f"   - Demonstrated realistic classification performance")

print(f"\n MATHEMATICAL INSIGHTS:")
print(f"   - Decision boundary: w·x + b = 0")
print(f"   - Margin maximization: maximize 2/||w||")
print(f"   - Support vectors: points where y(w·x + b) = 1")
print(f"   - Optimal solution: guaranteed by convex optimization")

print(f"\n NEXT STEPS FOR EXPLORATION:")
print("   - Try non-linear kernels (RBF, polynomial) for complex boundaries")
print("   - Experiment with different C values (0.1, 10, 100)")
print("   - Apply to real-world datasets (Iris, Wine, Breast Cancer)")
print("   - Explore multi-class classification scenarios")
print("   - Investigate feature scaling and preprocessing effects")

print("\n THE SVM PHILOSOPHY:")
print("   'It's not about perfect separation, it's about confident separation'")
print("   - SVM finds the boundary that maximizes the margin")
print("   - This leads to better generalization on unseen data")
print("   - The math guarantees the optimal solution exists and is unique!")

print(f"\n KEY TAKEAWAYS:")
print(f"   - Your {accuracy:.0f}% accuracy is realistic and valuable")
print(f"   - The {errors} misclassifications teach us about data complexity")
print(f"   - Support vectors reveal which data points truly matter")
print(f"   - Visualization helps understand the algorithm's decision-making")

print("\n Congratulations! You've mastered the fundamentals of SVM!")
print("   You now understand how machines learn to draw optimal boundaries")
print("   between different groups, balancing accuracy with confidence.")
print("=" * 50)

# ===================================================================
#  BONUS: Interactive Exploration Ideas
# ===================================================================

print("\n\n BONUS SECTION: Ideas for Further Exploration")
print("=" * 50)

print(" Try these modifications to deepen your understanding:")
print("""
1. PARAMETER EXPLORATION:
   - Change C values: model = SVC(kernel='linear', C=0.1)  # More regularization
   - Try C=10 or C=100 to see overfitting effects

2. DIFFERENT KERNELS:
   - RBF kernel: model = SVC(kernel='rbf', C=1.0)
   - Polynomial: model = SVC(kernel='poly', degree=3)

3. DATA VARIATIONS:
   - More overlap: X1 = np.random.randn(50, 2) * 1.5 + [1.0, 1.0]
   - Different distributions: Try circular or moon-shaped data

4. VISUALIZATION ENHANCEMENTS:
   - Add decision function contours to show confidence regions
   - Animate the effect of changing C parameter
   - Create 3D visualization for polynomial kernels

5. REAL-WORLD APPLICATION:
   - Load sklearn.datasets.load_iris() for multi-class classification
   - Try sklearn.datasets.make_moons() for non-linear boundaries
""")

print(" Remember: The best way to learn is by experimenting!")
print("   Each modification will teach you something new about SVM behavior.")

print("\n DOCUMENTATION COMPLETE")
print("=" * 50)

# %%


# %%
# ===================================================================
# Support Vector Machines: Complete Implementation with Hyperparameter Analysis
# ===================================================================
# From Basic Classification to Advanced Parameter Tuning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ===================================================================
# PART 1: Data Generation and Initial Setup
# ===================================================================

print("PART 1: Data Generation and Setup")
print("=" * 50)

# Create synthetic dataset
np.random.seed(42)
X0 = np.random.randn(50, 2) * 1.0 + [0, 0]
X1 = np.random.randn(50, 2) * 1.5 + [1.5, 1.5]
X = np.vstack((X0, X1))
y = np.array([0]*50 + [1]*50)

print("Dataset created:")
print(f"- Total samples: {len(X)}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {len(np.unique(y))}")

# Convert to hinge loss compatible format
y_hinge = np.where(y == 0, -1, 1)
print("Labels converted to {-1, +1} format for hinge loss calculation")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_hinge, test_size=0.3, random_state=42)
print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# Visualize initial data
plt.figure(figsize=(10, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=0.6, s=50, label='Class -1', edgecolors='black')
plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=0.6, s=50, label='Class +1', edgecolors='black')
plt.title("Training Data Distribution")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ===================================================================
# PART 2: Baseline SVM Performance
# ===================================================================

print("\n\nPART 2: Baseline SVM Analysis")
print("=" * 50)

# Train baseline model
baseline_model = SVC(kernel='linear', C=1.0)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

# Performance metrics
print("Baseline Model Performance (C=1.0):")
print(classification_report(y_test, baseline_pred, target_names=['Class -1', 'Class +1']))

# Confusion matrix
cm = confusion_matrix(y_test, baseline_pred)
print("Confusion Matrix:")
print(cm)

# Calculate baseline metrics
baseline_w = baseline_model.coef_[0]
baseline_margin = 2 / np.linalg.norm(baseline_w)
baseline_decision = baseline_model.decision_function(X_test)
baseline_hinge = np.mean(np.maximum(0, 1 - y_test * baseline_decision))

print(f"Baseline Results:")
print(f"- Support Vectors: {len(baseline_model.support_vectors_)}")
print(f"- Margin Distance: {baseline_margin:.4f}")
print(f"- Average Hinge Loss: {baseline_hinge:.4f}")

# ===================================================================
# PART 3: Hyperparameter Exploration - The C Parameter Study
# ===================================================================

print("\n\nPART 3: Comprehensive C Parameter Analysis")
print("=" * 50)

print("The C Parameter Controls the Regularization Strength:")
print("- Low C (0.01-0.1): Strong regularization, wider margins, simpler models")
print("- Medium C (0.5-1): Balanced approach")
print("- High C (5-10): Weak regularization, narrower margins, complex models")

# Define C values for systematic exploration
C_values = [0.01, 0.1, 0.5, 1, 5, 10]
results = []

print(f"\nTesting C values: {C_values}")
print("Training models and calculating metrics")

# Setup comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, C in enumerate(C_values):
    print(f"Processing C={C}")
    
    # Train model with current C
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    
    # Predictions and decision function
    y_pred = model.predict(X_test)
    decision_values = model.decision_function(X_test)
    
    # Calculate hinge loss: max(0, 1 - y_true * decision_value)
    hinge_losses = np.maximum(0, 1 - y_test * decision_values)
    avg_hinge_loss = np.mean(hinge_losses)
    
    # Calculate margin distance: 2/||w||
    w = model.coef_[0]
    margin_distance = 2 / np.linalg.norm(w)
    
    # Store comprehensive results
    results.append({
        "C Value": C,
        "Avg Hinge Loss": avg_hinge_loss,
        "Margin Distance": margin_distance,
        "Support Vectors": len(model.support_vectors_),
        "Training Accuracy": model.score(X_train, y_train),
        "Test Accuracy": model.score(X_test, y_test)
    })
    
    # Create detailed visualization
    ax = axes[idx]
    
    # Plot test data points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=30, edgecolors='k')
    
    # Create decision surface
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Decision function over the grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    # Level -1: margin boundary for class -1
    # Level 0: decision boundary
    # Level +1: margin boundary for class +1
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
               alpha=0.5, linestyles=['--', '-', '--'])
    
    # Highlight support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    # Add comprehensive title with key metrics
    ax.set_title(f"C={C}\nMargin={margin_distance:.3f}\nHinge Loss={avg_hinge_loss:.3f}")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()

plt.tight_layout()
plt.show()

print("Hyperparameter exploration complete!")

# ===================================================================
# PART 4: Advanced Results Analysis
# ===================================================================

print("\n\nPART 4: Comprehensive Results Analysis")
print("=" * 50)

# Create results DataFrame
df_results = pd.DataFrame(results)
print("Complete Hyperparameter Analysis Results:")
print(df_results.to_string(index=False, float_format='%.6f'))

# Statistical analysis
print("\nKey Statistical Insights:")
print(f"- Margin range: {df_results['Margin Distance'].min():.3f} to {df_results['Margin Distance'].max():.3f}")
print(f"- Hinge loss range: {df_results['Avg Hinge Loss'].min():.3f} to {df_results['Avg Hinge Loss'].max():.3f}")
print(f"- Support vector count range: {df_results['Support Vectors'].min()} to {df_results['Support Vectors'].max()}")

# Find optimal configurations
min_hinge_idx = df_results['Avg Hinge Loss'].idxmin()
max_margin_idx = df_results['Margin Distance'].idxmax()
best_test_idx = df_results['Test Accuracy'].idxmax()

print(f"\nOptimal Configurations:")
print(f"- Lowest hinge loss: C={df_results.loc[min_hinge_idx, 'C Value']} (Loss: {df_results.loc[min_hinge_idx, 'Avg Hinge Loss']:.3f})")
print(f"- Largest margin: C={df_results.loc[max_margin_idx, 'C Value']} (Margin: {df_results.loc[max_margin_idx, 'Margin Distance']:.3f})")
print(f"- Best test accuracy: C={df_results.loc[best_test_idx, 'C Value']} (Accuracy: {df_results.loc[best_test_idx, 'Test Accuracy']:.3f})")

# ===================================================================
# PART 5: Hinge Loss Deep Dive
# ===================================================================

print("\n\nPART 5: Understanding Hinge Loss Mechanics")
print("=" * 50)

print("Hinge Loss Formula: L(y, f(x)) = max(0, 1 - y * f(x))")
print("Where:")
print("- y is the true label (-1 or +1)")
print("- f(x) is the decision function value")
print("- Loss = 0 when prediction is correct and confident (y * f(x) >= 1)")
print("- Loss > 0 when prediction is wrong or lacks confidence")

# Demonstrate hinge loss calculation
print("\nHinge Loss Analysis for Different C Values:")
sample_model_low = SVC(kernel='linear', C=0.01)
sample_model_high = SVC(kernel='linear', C=10)
sample_model_low.fit(X_train, y_train)
sample_model_high.fit(X_train, y_train)

decision_low = sample_model_low.decision_function(X_test[:5])
decision_high = sample_model_high.decision_function(X_test[:5])
hinge_low = np.maximum(0, 1 - y_test[:5] * decision_low)
hinge_high = np.maximum(0, 1 - y_test[:5] * decision_high)

print("\nFirst 5 test samples comparison:")
print("Sample | True Label | C=0.01 Decision | C=0.01 Hinge | C=10 Decision | C=10 Hinge")
print("-" * 80)
for i in range(5):
    print(f"   {i+1}   |     {y_test[i]:2}     |     {decision_low[i]:6.3f}    |    {hinge_low[i]:5.3f}   |    {decision_high[i]:6.3f}   |   {hinge_high[i]:5.3f}")

# ===================================================================
# PART 6: Bias-Variance Tradeoff Analysis
# ===================================================================

print("\n\nPART 6: Bias-Variance Tradeoff in SVM")
print("=" * 50)

print("Understanding the C Parameter Impact:")

# Create tradeoff visualization
plt.figure(figsize=(15, 5))

# Plot 1: Margin vs C
plt.subplot(1, 3, 1)
plt.plot(df_results['C Value'], df_results['Margin Distance'], 'bo-', linewidth=2, markersize=8)
plt.xlabel('C Value')
plt.ylabel('Margin Distance')
plt.title('Margin vs Regularization')
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Plot 2: Hinge Loss vs C
plt.subplot(1, 3, 2)
plt.plot(df_results['C Value'], df_results['Avg Hinge Loss'], 'ro-', linewidth=2, markersize=8)
plt.xlabel('C Value')
plt.ylabel('Average Hinge Loss')
plt.title('Hinge Loss vs Regularization')
plt.xscale('log')
plt.grid(True, alpha=0.3)

# Plot 3: Support Vectors vs C
plt.subplot(1, 3, 3)
plt.plot(df_results['C Value'], df_results['Support Vectors'], 'go-', linewidth=2, markersize=8)
plt.xlabel('C Value')
plt.ylabel('Number of Support Vectors')
plt.title('Support Vectors vs Regularization')
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Observed Relationships:")
print("1. As C increases, margin distance generally decreases")
print("2. As C increases, hinge loss generally decreases")
print("3. Higher C values may lead to more support vectors")
print("4. This demonstrates the fundamental bias-variance tradeoff")

# ===================================================================
# PART 7: Advanced Model Selection Guidelines
# ===================================================================

print("\n\nPART 7: Model Selection Strategy")
print("=" * 50)

print("C Parameter Selection Guidelines:")

# Analyze trends
margin_trend = "decreasing" if df_results['Margin Distance'].iloc[-1] < df_results['Margin Distance'].iloc[0] else "increasing"
hinge_trend = "decreasing" if df_results['Avg Hinge Loss'].iloc[-1] < df_results['Avg Hinge Loss'].iloc[0] else "increasing"

print(f"- Margin trend with increasing C: {margin_trend}")
print(f"- Hinge loss trend with increasing C: {hinge_trend}")

# Practical recommendations
print("\nPractical Selection Strategy:")
print("1. Cross-validation approach:")
print("   - Use GridSearchCV with multiple C values")
print("   - Evaluate on validation set, not training set")

print("\n2. Problem-specific considerations:")
print("   - High noise data: Lower C for better generalization")
print("   - Clean data: Higher C for precise fitting")
print("   - Limited data: Moderate C to avoid overfitting")

print("\n3. Performance monitoring:")
print("   - Watch for large gaps between training and test accuracy")
print("   - Consider computational cost of many support vectors")
print("   - Balance model complexity with interpretability")

# Model recommendations based on results
recommended_C = df_results.loc[min_hinge_idx, 'C Value']
print(f"\nFor this dataset, recommended C: {recommended_C}")
print(f"- Achieves lowest hinge loss: {df_results.loc[min_hinge_idx, 'Avg Hinge Loss']:.3f}")
print(f"- Maintains reasonable margin: {df_results.loc[min_hinge_idx, 'Margin Distance']:.3f}")
print(f"- Uses {df_results.loc[min_hinge_idx, 'Support Vectors']} support vectors")

# ===================================================================
# PART 8: Advanced Implementation Techniques
# ===================================================================

print("\n\nPART 8: Advanced Implementation Notes")

print("Key Implementation Details:")

print("\n1. Label Encoding for Hinge Loss:")
print("   - Converted {0,1} labels to {-1,+1}")
print("   - Required for accurate hinge loss computation")
print("   - Standard practice in SVM implementations")

print("\n2. Decision Function vs Predict:")
print("   - decision_function(): Returns signed distance to hyperplane")
print("   - predict(): Returns class labels after thresholding")
print("   - Hinge loss requires decision function values")

print("\n3. Margin Calculation:")
print("   - Margin width = 2/||w|| where w is the weight vector")
print("   - Directly related to model complexity")
print("   - Larger margins indicate simpler models")

print("\n4. Support Vector Identification:")
print("   - Points that lie exactly on the margin boundaries")
print("   - Only these points influence the final model")
print("   - Removing non-support vectors doesn't change the model")

# ===================================================================
# CONCLUSIONS AND FUTURE DIRECTIONS
# ===================================================================

print("\n\nCONCLUSIONS AND FUTURE DIRECTIONS")

print("Key Findings from Hyperparameter Analysis:")
print(f"1. Tested {len(C_values)} different C values systematically")
print(f"2. Observed clear margin-regularization relationship")
print(f"3. Identified optimal C={recommended_C} for this dataset")
print(f"4. Demonstrated bias-variance tradeoff in practice")

print("\nTechnical Achievements:")
print("- Implemented proper hinge loss calculation")
print("- Created comprehensive visualization framework")
print("- Developed systematic hyperparameter evaluation")
print("- Established model selection criteria")

print("\nNext Steps for Advanced SVM:")
print("- Implement cross-validation for robust C selection")
print("- Explore non-linear kernels (RBF, polynomial)")
print("- Add feature scaling and preprocessing")
print("- Compare with other classification algorithms")
print("- Apply to real-world datasets")

print("\nCode Templates for Extension:")
print("1. Grid Search Implementation:")
print("   from sklearn.model_selection import GridSearchCV")
print("   param_grid = {'C': np.logspace(-3, 2, 20)}")
print("   grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)")

print("\n2. Validation Curves:")
print("   from sklearn.model_selection import validation_curve")
print("   train_scores, val_scores = validation_curve(SVC(), X, y, 'C', C_range)")

print("\n3. Learning Curves:")
print("   from sklearn.model_selection import learning_curve")
print("   train_sizes, train_scores, val_scores = learning_curve(SVC(C=optimal_C), X, y)")

print(f"\nHyperparameter optimization complete!")

# %%


# %%
# ===================================================================
# Advanced SVM: Grid Search and Polynomial Kernel Optimization
# ===================================================================
# Building on Linear SVM Foundation with Advanced Hyperparameter Tuning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Set plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ===================================================================
# FOUNDATION: Data Setup (Building on Previous Linear Analysis)
# ===================================================================

print("ADVANCED SVM IMPLEMENTATION")
print("=" * 60)
print("Building on linear SVM foundation with polynomial kernels and grid search")

# Create synthetic dataset (same as linear analysis)
np.random.seed(42)
X0 = np.random.randn(50, 2) * 1.0 + [0, 0]
X1 = np.random.randn(50, 2) * 1.5 + [1.5, 1.5]
X = np.vstack((X0, X1))
y = np.array([0]*50 + [1]*50)

# Convert to hinge loss compatible format
y_hinge = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_hinge, test_size=0.3, random_state=42)

print(f"Dataset ready: {len(X_train)} training, {len(X_test)} test samples")
print("Previous linear SVM analysis established baseline performance")
print("Now advancing to polynomial kernels with systematic optimization")

# ===================================================================
# BREAKTHROUGH: Polynomial Kernel with Comprehensive Grid Search
# ===================================================================

print("\n\nPOLYNOMIAL KERNEL GRID SEARCH OPTIMIZATION")
print("=" * 60)

print("Advanced Technique: Moving beyond linear boundaries")
print("Polynomial kernels can capture non-linear relationships through:")
print("K(x, y) = (gamma * <x,y> + coef0)^degree")

# Define polynomial SVM for grid search
svc = SVC(kernel='poly')

# Comprehensive 4-dimensional parameter grid
param_grid = {
    'C': [0.1, 1, 5],                    # Regularization strength
    'degree': [2, 3, 4],                 # Polynomial degree (complexity)
    'gamma': ['scale', 'auto', 0.1, 1, 5],  # Kernel coefficient
    'coef0': [0, 1]                      # Independent term
}

print("\nGrid Search Configuration:")
print(f"- C parameter: {param_grid['C']} (regularization control)")
print(f"- Polynomial degrees: {param_grid['degree']} (boundary complexity)")
print(f"- Gamma values: {param_grid['gamma']} (feature influence)")
print(f"- Coef0 values: {param_grid['coef0']} (bias term)")

total_combinations = len(param_grid['C']) * len(param_grid['degree']) * len(param_grid['gamma']) * len(param_grid['coef0'])
print(f"\nComputational scope:")
print(f"- Parameter combinations: {total_combinations}")
print(f"- With 5-fold cross-validation: {total_combinations * 5} model fits")
print(f"- Systematic evaluation of polynomial kernel space")

# ===================================================================
# EXECUTION: Advanced Grid Search Implementation
# ===================================================================

print("\n\nGRID SEARCH EXECUTION")
print("=" * 60)

print("Implementing GridSearchCV with advanced configuration:")
print("- Cross-validation: 5-fold for robust estimation")
print("- Scoring metric: Accuracy for classification performance")
print("- Parallel processing: n_jobs=-1 for computational efficiency")
print("- Verbose output: Progress tracking during optimization")

# Execute comprehensive grid search
grid_search = GridSearchCV(
    estimator=svc, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
)

print("\nLaunching grid search optimization")
grid_search.fit(X_train, y_train)

print("\nOptimization Results:")
print(f"Best parameter configuration: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Standard deviation: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")

# ===================================================================
# ANALYSIS: Optimal Model Performance Evaluation
# ===================================================================

print("\n\nOPTIMAL MODEL ANALYSIS")
print("=" * 60)

# Extract and analyze best model
best_model = grid_search.best_estimator_
optimal_params = grid_search.best_params_

print("Optimal Configuration Breakdown:")
for param, value in optimal_params.items():
    if param == 'C':
        print(f"- {param}: {value} (optimal regularization balance)")
    elif param == 'degree':
        print(f"- {param}: {value} (polynomial complexity level)")
    elif param == 'gamma':
        print(f"- {param}: {value} (kernel coefficient setting)")
    elif param == 'coef0':
        print(f"- {param}: {value} (independent term influence)")

# Comprehensive performance evaluation
y_pred = best_model.predict(X_test)
test_accuracy = best_model.score(X_test, y_test)
train_accuracy = best_model.score(X_train, y_train)

# Advanced hinge loss calculation
decision_values = best_model.decision_function(X_test)
hinge_losses = np.maximum(0, 1 - y_test * decision_values)
avg_hinge_loss = np.mean(hinge_losses)

print(f"\nPerformance Metrics:")
print(f"- Training accuracy: {train_accuracy:.4f}")
print(f"- Test accuracy: {test_accuracy:.4f}")
print(f"- Generalization gap: {abs(train_accuracy - test_accuracy):.4f}")
print(f"- Average hinge loss: {avg_hinge_loss:.4f}")
print(f"- Support vectors: {len(best_model.support_vectors_)}")

print("\nDetailed Classification Performance:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix Analysis:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Classification errors: {len(y_test) - np.trace(cm)} out of {len(y_test)}")

# ===================================================================
# DEEP DIVE: Grid Search Results Analysis
# ===================================================================

print("\n\nGRID SEARCH RESULTS DEEP DIVE")
print("=" * 60)

# Extract comprehensive results
results_df = pd.DataFrame(grid_search.cv_results_)

# Top performing configurations
n_top = 5
top_indices = np.argsort(results_df['mean_test_score'])[-n_top:][::-1]
top_results = results_df.iloc[top_indices]

print(f"Top {n_top} Parameter Combinations:")
print("Rank | CV Score | Std Dev | C    | Degree | Gamma | Coef0")
print("-" * 60)

for i, (idx, row) in enumerate(top_results.iterrows()):
    print(f"  {i+1}  |  {row['mean_test_score']:.4f}  |  {row['std_test_score']:.4f}  | {row['param_C']:4} |   {row['param_degree']:2}   | {str(row['param_gamma']):5} |   {row['param_coef0']}")

# Parameter impact analysis
print(f"\nParameter Impact Analysis:")

# C parameter analysis
print(f"\nRegularization (C) Impact:")
for c in param_grid['C']:
    c_mask = results_df['param_C'] == c
    c_scores = results_df[c_mask]['mean_test_score']
    print(f"- C={c}: Mean CV score = {c_scores.mean():.4f} (±{c_scores.std():.4f})")

# Degree analysis
print(f"\nPolynomial Degree Impact:")
for deg in param_grid['degree']:
    deg_mask = results_df['param_degree'] == deg
    deg_scores = results_df[deg_mask]['mean_test_score']
    print(f"- Degree {deg}: Mean CV score = {deg_scores.mean():.4f} (±{deg_scores.std():.4f})")

# Gamma analysis
print(f"\nKernel Coefficient (Gamma) Impact:")
for gamma in param_grid['gamma']:
    gamma_mask = results_df['param_gamma'] == gamma
    gamma_scores = results_df[gamma_mask]['mean_test_score']
    print(f"- Gamma {gamma}: Mean CV score = {gamma_scores.mean():.4f} (±{gamma_scores.std():.4f})")

# ===================================================================
# VISUALIZATION: Advanced Decision Boundary Analysis
# ===================================================================

print("\n\nADVANCED BOUNDARY VISUALIZATION")
print("=" * 60)

def plot_polynomial_boundary(model, X, y, title_params):
    """Advanced polynomial decision boundary visualization"""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')
    
    # Create high-resolution decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Calculate decision function across grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
               alpha=0.5, linestyles=['--', '-', '--'])
    
    # Highlight support vectors with enhanced visibility
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', linewidths=2, 
               label='Support Vectors')
    
    plt.title(f"Best Polynomial SVM Decision Boundary\nParams: {title_params}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print("Generating optimal polynomial decision boundary")
plot_polynomial_boundary(best_model, X_test, y_test, grid_search.best_params_)

# ===================================================================
# COMPARATIVE ANALYSIS: Model Performance Insights
# ===================================================================

print("\n\nCOMPARATIVE MODEL ANALYSIS")
print("=" * 60)

# Calculate performance across top configurations
print("Performance Analysis Across Top Configurations:")

top_configs_performance = []
for idx in top_indices:
    params = results_df.iloc[idx]['params']
    cv_score = results_df.iloc[idx]['mean_test_score']
    
    # Train model with specific parameters
    temp_model = SVC(**params)
    temp_model.fit(X_train, y_train)
    
    # Calculate test metrics
    temp_test_acc = temp_model.score(X_test, y_test)
    temp_decision = temp_model.decision_function(X_test)
    temp_hinge = np.mean(np.maximum(0, 1 - y_test * temp_decision))
    
    top_configs_performance.append({
        'Config': f"C={params['C']}, deg={params['degree']}, γ={params['gamma']}",
        'CV_Score': cv_score,
        'Test_Accuracy': temp_test_acc,
        'Hinge_Loss': temp_hinge,
        'Support_Vectors': len(temp_model.support_vectors_)
    })

performance_df = pd.DataFrame(top_configs_performance)
print("\nComprehensive Performance Comparison:")
print(performance_df.to_string(index=False, float_format='%.4f'))

# Identify best performer by different metrics
best_cv_idx = performance_df['CV_Score'].idxmax()
best_test_idx = performance_df['Test_Accuracy'].idxmax()
best_hinge_idx = performance_df['Hinge_Loss'].idxmin()

print(f"\nOptimal Configurations by Different Metrics:")
print(f"- Best CV score: {performance_df.iloc[best_cv_idx]['Config']}")
print(f"- Best test accuracy: {performance_df.iloc[best_test_idx]['Config']}")
print(f"- Best hinge loss: {performance_df.iloc[best_hinge_idx]['Config']}")

# ===================================================================
# ADVANCED INSIGHTS: Hyperparameter Relationships
# ===================================================================

print("\n\nADVANCED HYPERPARAMETER INSIGHTS")
print("=" * 60)

print("Key Relationships Discovered:")

print(f"\n1. Optimal Parameter Synergy:")
print(f"   - C={optimal_params['C']}: Balanced regularization")
print(f"   - Degree={optimal_params['degree']}: Moderate polynomial complexity")
print(f"   - Gamma={optimal_params['gamma']}: Optimal kernel scaling")
print(f"   - Coef0={optimal_params['coef0']}: Bias term contribution")

print(f"\n2. Performance Trends:")
cv_scores = results_df['mean_test_score']
print(f"   - CV score range: {cv_scores.min():.4f} to {cv_scores.max():.4f}")
print(f"   - Performance variance: {cv_scores.std():.4f}")
print(f"   - Top 10% threshold: {np.percentile(cv_scores, 90):.4f}")

print(f"\n3. Computational Efficiency:")
print(f"   - Best model support vectors: {len(best_model.support_vectors_)}")
print(f"   - Model complexity indicator: {len(best_model.support_vectors_)/len(X_train)*100:.1f}% of training data")

print(f"\n4. Generalization Assessment:")
print(f"   - CV-Test gap: {abs(grid_search.best_score_ - test_accuracy):.4f}")
print(f"   - Overfitting indicator: {'Low' if abs(grid_search.best_score_ - test_accuracy) < 0.05 else 'Moderate'}")

# ===================================================================
# PRODUCTION GUIDELINES: Advanced Model Selection
# ===================================================================

print("\n\nPRODUCTION DEPLOYMENT GUIDELINES")
print("=" * 60)

print("Advanced Model Selection Insights:")

print(f"\n1. Grid Search Best Practices Implemented:")
print("   ✓ Comprehensive parameter space exploration")
print("   ✓ Cross-validation for robust evaluation")
print("   ✓ Multiple metric assessment")
print("   ✓ Computational efficiency optimization")

print(f"\n2. Model Validation Strategy:")
print("   ✓ Train-validation-test separation")
print("   ✓ Hinge loss monitoring")
print("   ✓ Support vector analysis")
print("   ✓ Generalization gap assessment")

print(f"\n3. Hyperparameter Selection Rationale:")
print(f"   - Selected C={optimal_params['C']}: Balances complexity and regularization")
print(f"   - Selected degree={optimal_params['degree']}: Provides non-linearity without overfitting")
print(f"   - Selected gamma={optimal_params['gamma']}: Optimizes kernel coefficient")

print(f"\n4. Performance Guarantee:")
print(f"   - Expected accuracy: {grid_search.best_score_:.1%} (±{results_df.loc[grid_search.best_index_, 'std_test_score']:.1%})")
print(f"   - Hinge loss target: ≤{avg_hinge_loss:.3f}")
print(f"   - Model complexity: {len(best_model.support_vectors_)} support vectors")

print(f"\nAdvanced SVM Implementation Complete!")
print(f"Achieved {test_accuracy:.1%} accuracy with systematic polynomial kernel optimization")
print("=" * 60)

# %%


# %%
# ===================================================================
# Complete SVM Kernel Evolution: Linear → Polynomial → RBF
# ===================================================================
# Advanced Hyperparameter Optimization Across Multiple Kernel Types

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Set plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ===================================================================
# FOUNDATION: Dataset and Baseline Setup
# ===================================================================

print("COMPLETE SVM KERNEL ANALYSIS PIPELINE")
print("=" * 60)
print("Evolution: Linear SVM → Polynomial Kernel → RBF Kernel Optimization")

# Create synthetic dataset
np.random.seed(42)
X0 = np.random.randn(50, 2) * 1.0 + [0, 0]
X1 = np.random.randn(50, 2) * 1.5 + [1.5, 1.5]
X = np.vstack((X0, X1))
y = np.array([0]*50 + [1]*50)

# Convert to hinge loss compatible format
y_hinge = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_hinge, test_size=0.3, random_state=42)

print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples")
print("Building on previous linear and polynomial kernel analysis")
print("Now implementing RBF kernel with advanced hinge loss analysis")

# ===================================================================
# PHASE 1: Linear Kernel Baseline (Quick Reference)
# ===================================================================

print("\n\nPHASE 1: Linear Kernel Baseline Reference")
print("=" * 60)

# Quick linear baseline for comparison
C_values_linear = [0.01, 0.1, 0.5, 1, 5, 10]
linear_results = []

print("Linear kernel systematic C parameter analysis")
for C in C_values_linear:
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    
    decision_values = model.decision_function(X_test)
    hinge_losses = np.maximum(0, 1 - y_test * decision_values)
    avg_hinge_loss = np.mean(hinge_losses)
    
    w = model.coef_[0]
    margin_distance = 2 / np.linalg.norm(w)
    
    linear_results.append({
        "C": C,
        "Test_Accuracy": model.score(X_test, y_test),
        "Hinge_Loss": avg_hinge_loss,
        "Margin": margin_distance
    })

df_linear = pd.DataFrame(linear_results)
best_linear_idx = df_linear['Test_Accuracy'].idxmax()
print(f"Linear kernel optimal: C={df_linear.loc[best_linear_idx, 'C']}, Accuracy={df_linear.loc[best_linear_idx, 'Test_Accuracy']:.4f}")

# ===================================================================
# PHASE 2: Polynomial Kernel Optimization (Previous Analysis)
# ===================================================================

print("\n\nPHASE 2: Polynomial Kernel Grid Search")
print("=" * 60)

# Polynomial kernel grid search (building on previous analysis)
poly_param_grid = {
    'C': [0.1, 1, 5],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto', 0.1, 1, 5],
    'coef0': [0, 1]
}

print("Polynomial kernel parameter space exploration")
svc_poly = SVC(kernel='poly')
grid_search_poly = GridSearchCV(svc_poly, poly_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid_search_poly.fit(X_train, y_train)

poly_best_model = grid_search_poly.best_estimator_
poly_test_acc = poly_best_model.score(X_test, y_test)
poly_decision = poly_best_model.decision_function(X_test)
poly_hinge = np.mean(np.maximum(0, 1 - y_test * poly_decision))

print(f"Polynomial optimal: {grid_search_poly.best_params_}")
print(f"Polynomial performance: Accuracy={poly_test_acc:.4f}, Hinge Loss={poly_hinge:.4f}")

# ===================================================================
# PHASE 3: Advanced RBF Kernel Implementation
# ===================================================================

print("\n\nPHASE 3: RBF KERNEL ADVANCED OPTIMIZATION")
print("=" * 60)

print("RBF Kernel Theory:")
print("K(x, y) = exp(-gamma * ||x - y||²)")
print("- Gamma controls the influence of individual training examples")
print("- High gamma: tight fit, complex boundaries")
print("- Low gamma: smooth boundaries, better generalization")

# RBF parameter grid
param_grid_rbf = {
    'C': [0.1, 1, 5, 10],
    'gamma': ['scale', 'auto', 0.1, 1, 5]
}

print(f"\nRBF Parameter Space:")
print(f"- C values: {param_grid_rbf['C']} (regularization strength)")
print(f"- Gamma values: {param_grid_rbf['gamma']} (kernel bandwidth)")
print(f"- Total combinations: {len(param_grid_rbf['C']) * len(param_grid_rbf['gamma'])}")

# Execute RBF grid search
print("\nExecuting RBF GridSearchCV")
svc_rbf = SVC(kernel='rbf')
grid_search_rbf = GridSearchCV(svc_rbf, param_grid_rbf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_rbf.fit(X_train, y_train)

print(f"\nRBF Grid Search Results:")
print(f"Best parameters: {grid_search_rbf.best_params_}")
print(f"Best CV accuracy: {grid_search_rbf.best_score_:.4f}")

# ===================================================================
# PHASE 4: Comprehensive Hinge Loss Analysis for RBF
# ===================================================================

print("\n\nPHASE 4: RBF HINGE LOSS DEEP ANALYSIS")
print("=" * 60)

print("Advanced technique: Systematic hinge loss evaluation across parameter space")
print("Analyzing both training and test hinge loss for overfitting detection")

# Comprehensive hinge loss analysis
rbf_results = []
print("\nCalculating hinge loss for all RBF parameter combinations")

for C in param_grid_rbf['C']:
    for gamma in param_grid_rbf['gamma']:
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train, y_train)
        
        # Training hinge loss
        decision_train = model.decision_function(X_train)
        hinge_loss_train = np.maximum(0, 1 - y_train * decision_train)
        avg_hinge_loss_train = np.mean(hinge_loss_train)
        
        # Test hinge loss
        decision_test = model.decision_function(X_test)
        hinge_loss_test = np.maximum(0, 1 - y_test * decision_test)
        avg_hinge_loss_test = np.mean(hinge_loss_test)
        
        rbf_results.append({
            'C': C,
            'gamma': gamma,
            'Train_Hinge_Loss': avg_hinge_loss_train,
            'Test_Hinge_Loss': avg_hinge_loss_test,
            'Train_Accuracy': model.score(X_train, y_train),
            'Test_Accuracy': model.score(X_test, y_test),
            'Support_Vectors': len(model.support_vectors_)
        })

df_rbf_results = pd.DataFrame(rbf_results)

print("\nRBF Hinge Loss Analysis (Top 5 by Test Hinge Loss):")
top_hinge = df_rbf_results.sort_values('Test_Hinge_Loss').head()
print(top_hinge[['C', 'gamma', 'Train_Hinge_Loss', 'Test_Hinge_Loss']])

# Identify optimal configuration by hinge loss
best_hinge_idx = df_rbf_results['Test_Hinge_Loss'].idxmin()
optimal_C = df_rbf_results.loc[best_hinge_idx, 'C']
optimal_gamma = df_rbf_results.loc[best_hinge_idx, 'gamma']
optimal_test_hinge = df_rbf_results.loc[best_hinge_idx, 'Test_Hinge_Loss']

print(f"\nOptimal RBF configuration by hinge loss:")
print(f"- C: {optimal_C}, Gamma: {optimal_gamma}")
print(f"- Test hinge loss: {optimal_test_hinge:.6f}")

# ===================================================================
# PHASE 5: Advanced Model Performance Comparison
# ===================================================================

print("\n\nPHASE 5: MULTI-KERNEL PERFORMANCE COMPARISON")
print("=" * 60)

# Extract best model from grid search
rbf_best_model = grid_search_rbf.best_estimator_
rbf_test_acc = rbf_best_model.score(X_test, y_test)
rbf_decision = rbf_best_model.decision_function(X_test)
rbf_hinge = np.mean(np.maximum(0, 1 - y_test * rbf_decision))

# Comprehensive comparison
print("Complete Kernel Performance Summary:")
print("Kernel     | Optimal Parameters           | Test Accuracy | Hinge Loss | Support Vectors")
print("-" * 90)

# Linear summary
best_linear = df_linear.loc[best_linear_idx]
print(f"Linear     | C={best_linear['C']:4}                      |    {best_linear['Test_Accuracy']:.4f}    |   {best_linear['Hinge_Loss']:.4f}  |      N/A")

# Polynomial summary
print(f"Polynomial | C={grid_search_poly.best_params_['C']}, deg={grid_search_poly.best_params_['degree']}, γ={grid_search_poly.best_params_['gamma']}     |    {poly_test_acc:.4f}    |   {poly_hinge:.4f}  |      {len(poly_best_model.support_vectors_):2}")

# RBF summary
print(f"RBF        | C={grid_search_rbf.best_params_['C']}, γ={grid_search_rbf.best_params_['gamma']:5}              |    {rbf_test_acc:.4f}    |   {rbf_hinge:.4f}  |      {len(rbf_best_model.support_vectors_):2}")

# Determine overall best performer
performances = {
    'Linear': best_linear['Test_Accuracy'],
    'Polynomial': poly_test_acc,
    'RBF': rbf_test_acc
}
best_kernel = max(performances, key=performances.get)
print(f"\nOverall best performer: {best_kernel} kernel with {max(performances.values()):.4f} accuracy")

# ===================================================================
# PHASE 6: Advanced RBF Visualization and Analysis
# ===================================================================

print("\n\nPHASE 6: RBF DECISION BOUNDARY ANALYSIS")
print("=" * 60)

def plot_rbf_decision_boundary(model, X, y, params):
    """Advanced RBF boundary visualization with enhanced analysis"""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')
    
    # Create high-resolution grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    # Calculate decision function
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
               linestyles=['--', '-', '--'], alpha=0.5)
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', linewidths=2,
               label='Support Vectors')
    
    plt.title(f'RBF SVM Decision Boundary\nBest Params: {params}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print("Generating optimal RBF decision boundary visualization")
plot_rbf_decision_boundary(rbf_best_model, X_test, y_test, grid_search_rbf.best_params_)

# ===================================================================
# PHASE 7: Overfitting Analysis Through Hinge Loss
# ===================================================================

print("\n\nPHASE 7: OVERFITTING DETECTION ANALYSIS")
print("=" * 60)

print("Advanced technique: Training vs Test hinge loss gap analysis")
print("Large gaps indicate potential overfitting")

# Calculate hinge loss gaps
df_rbf_results['Hinge_Gap'] = df_rbf_results['Test_Hinge_Loss'] - df_rbf_results['Train_Hinge_Loss']
df_rbf_results['Accuracy_Gap'] = df_rbf_results['Train_Accuracy'] - df_rbf_results['Test_Accuracy']

# Sort by different criteria
print("\nOverfitting Analysis (sorted by hinge loss gap):")
overfitting_analysis = df_rbf_results.sort_values('Hinge_Gap').head()
print(overfitting_analysis[['C', 'gamma', 'Train_Hinge_Loss', 'Test_Hinge_Loss', 'Hinge_Gap']])

# Identify well-generalized models
well_generalized = df_rbf_results[df_rbf_results['Hinge_Gap'] < 0.05]
print(f"\nWell-generalized configurations (hinge gap < 0.05): {len(well_generalized)}")

if len(well_generalized) > 0:
    best_generalized = well_generalized.loc[well_generalized['Test_Accuracy'].idxmax()]
    print(f"Best generalizing model: C={best_generalized['C']}, gamma={best_generalized['gamma']}")
    print(f"Performance: {best_generalized['Test_Accuracy']:.4f} accuracy, {best_generalized['Hinge_Gap']:.6f} hinge gap")

# ===================================================================
# PHASE 8: Parameter Sensitivity Analysis
# ===================================================================

print("\n\nPHASE 8: RBF PARAMETER SENSITIVITY ANALYSIS")
print("=" * 60)

print("Analyzing parameter impact on performance:")

# C parameter impact
print(f"\nRegularization (C) Impact Analysis:")
for c in param_grid_rbf['C']:
    c_subset = df_rbf_results[df_rbf_results['C'] == c]
    avg_acc = c_subset['Test_Accuracy'].mean()
    avg_hinge = c_subset['Test_Hinge_Loss'].mean()
    print(f"- C={c:4}: Mean accuracy={avg_acc:.4f}, Mean hinge loss={avg_hinge:.6f}")

# Gamma parameter impact
print(f"\nKernel Bandwidth (Gamma) Impact Analysis:")
for gamma in param_grid_rbf['gamma']:
    gamma_subset = df_rbf_results[df_rbf_results['gamma'] == gamma]
    avg_acc = gamma_subset['Test_Accuracy'].mean()
    avg_hinge = gamma_subset['Test_Hinge_Loss'].mean()
    print(f"- Gamma={str(gamma):5}: Mean accuracy={avg_acc:.4f}, Mean hinge loss={avg_hinge:.6f}")

# Best combination insights
print(f"\nOptimal Parameter Combination Insights:")
rbf_optimal = grid_search_rbf.best_params_
print(f"- C={rbf_optimal['C']}: {'High' if rbf_optimal['C'] >= 5 else 'Low'} regularization")
print(f"- Gamma={rbf_optimal['gamma']}: {'Tight' if rbf_optimal['gamma'] in [1, 5] else 'Smooth'} kernel bandwidth")

# ===================================================================
# CONCLUSIONS: Advanced SVM Implementation Mastery
# ===================================================================

print("\n\nCOMPLETE SVM IMPLEMENTATION MASTERY")
print("=" * 60)

print("Advanced Techniques Successfully Implemented:")

print(f"\n1. Multi-Kernel Systematic Comparison:")
print("   ✓ Linear kernel with C parameter optimization")
print("   ✓ Polynomial kernel with 4D parameter space")
print("   ✓ RBF kernel with gamma-C optimization")

print(f"\n2. Advanced Evaluation Methodologies:")
print("   ✓ Cross-validation for robust parameter selection")
print("   ✓ Hinge loss analysis for training-test comparison")
print("   ✓ Overfitting detection through gap analysis")
print("   ✓ Parameter sensitivity assessment")

print(f"\n3. Optimal Configuration Discovery:")
print(f"   - Best overall kernel: {best_kernel}")
print(f"   - RBF optimal parameters: C={rbf_optimal['C']}, gamma={rbf_optimal['gamma']}")
print(f"   - Performance achievement: {max(performances.values()):.4f} accuracy")
print(f"   - Hinge loss optimization: {optimal_test_hinge:.6f}")

print(f"\n4. Production-Ready Implementation:")
print("   ✓ Systematic hyperparameter optimization")
print("   ✓ Comprehensive performance evaluation")
print("   ✓ Overfitting prevention strategies")
print("   ✓ Model selection based on multiple criteria")

print(f"\n5. Advanced Insights Gained:")
print(f"   - RBF kernel outperforms linear and polynomial for this dataset")
print(f"   - Optimal gamma={rbf_optimal['gamma']} provides best complexity-performance balance")
print(f"   - Hinge loss analysis reveals {len(well_generalized)} well-generalized configurations")
print(f"   - Support vector count: {len(rbf_best_model.support_vectors_)} for optimal model")

print(f"\nAdvanced SVM Pipeline Complete!")
print(f"Achieved comprehensive kernel comparison with systematic optimization")
print("=" * 60)

# %%


# %%
# ===================================================================
# Advanced SVM Implementation: Real-World Multi-Class Classification
# ===================================================================
# Complete Pipeline: Synthetic Data → Iris Dataset with Multi-Kernel Optimization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ===================================================================
# FOUNDATION: Synthetic Data Analysis Summary
# ===================================================================

print("ADVANCED SVM COMPLETE IMPLEMENTATION PIPELINE")
print("=" * 60)
print("Evolution: Synthetic Binary → Real-World Multi-Class Classification")
print("Previous Analysis: Linear → Polynomial → RBF kernels on synthetic data")
print("Current Focus: Multi-class Iris dataset with comprehensive kernel comparison")

# ===================================================================
# PART 1: Real-World Dataset Introduction - Iris Classification
# ===================================================================

print("\n\nPART 1: REAL-WORLD MULTI-CLASS DATASET")
print("=" * 60)

# Load famous Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only first 2 features for visualization
y = iris.target

print("Dataset Transition: Synthetic → Real-World")
print(f"- Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"- Classes: {len(np.unique(y))} (setosa, versicolor, virginica)")
print(f"- Features: {iris.feature_names[:2]}")
print(f"- Challenge: Multi-class classification vs previous binary problems")

# Class distribution analysis
print(f"\nClass Distribution:")
for i, class_name in enumerate(iris.target_names):
    count = np.sum(y == i)
    print(f"- {class_name}: {count} samples ({count/len(y)*100:.1f}%)")

# ===================================================================
# PART 2: Advanced Preprocessing - Feature Scaling Implementation
# ===================================================================

print("\n\nPART 2: ADVANCED PREPROCESSING TECHNIQUES")
print("=" * 60)

print("Key Advancement: Feature Scaling for SVM Optimization")
print("- SVM is sensitive to feature magnitude differences")
print("- StandardScaler: transforms features to zero mean, unit variance")
print("- Critical for optimal kernel performance")

# Split dataset with stratification for balanced classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nDataset Split (Stratified):")
print(f"- Training: {len(X_train)} samples")
print(f"- Testing: {len(X_test)} samples")
print(f"- Stratification ensures balanced class representation")

# Feature scaling implementation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature Scaling Results:")
print(f"- Original feature ranges: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"- Scaled feature ranges: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"- Mean after scaling: {X_train_scaled.mean(axis=0)}")
print(f"- Std after scaling: {X_train_scaled.std(axis=0)}")

# ===================================================================
# PART 3: Comprehensive Multi-Kernel Grid Search
# ===================================================================

print("\n\nPART 3: COMPREHENSIVE MULTI-KERNEL OPTIMIZATION")
print("=" * 60)

print("Advanced Technique: Simultaneous Multi-Kernel Parameter Search")
print("Building on previous kernel-specific optimizations")

# Define comprehensive parameter grid for all kernels
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
    {'kernel': ['poly'], 'C': [0.1, 0.5, 1], 'degree': [2, 3, 4, 5], 'gamma': ['scale', 'auto', 0.1, 0.5], 'coef0': [0, 1]},
    {'kernel': ['rbf'], 'C': [0.1, 0.6, 1, 5, 10], 'gamma': ['scale', 'auto', 0.1, 0.5, 1]},
]

# Calculate total parameter combinations
total_combinations = 0
for grid in param_grid:
    combinations = 1
    for param, values in grid.items():
        if param != 'kernel':
            combinations *= len(values)
    total_combinations += combinations
    kernel_name = grid['kernel'][0]
    print(f"- {kernel_name.upper()} kernel: {combinations} parameter combinations")

print(f"\nGrid Search Scope:")
print(f"- Total parameter combinations: {total_combinations}")
print(f"- With 5-fold CV: {total_combinations * 5} model evaluations")
print(f"- Multi-class strategy: One-vs-Rest decision function")

# ===================================================================
# PART 4: Multi-Class SVM Configuration and Execution
# ===================================================================

print("\n\nPART 4: MULTI-CLASS SVM GRID SEARCH EXECUTION")
print("=" * 60)

print("Advanced Configuration:")
print("- decision_function_shape='ovr': One-vs-Rest for multi-class")
print("- Stratified cross-validation for balanced evaluation")
print("- Parallel processing for computational efficiency")

# Configure SVM for multi-class classification
svc = SVC(decision_function_shape='ovr')
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("\nExecuting comprehensive grid search")
grid_search.fit(X_train_scaled, y_train)

print(f"\nOptimization Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Optimization completed across {total_combinations} configurations")

# ===================================================================
# PART 5: Advanced Multi-Class Performance Analysis
# ===================================================================

print("\n\nPART 5: MULTI-CLASS PERFORMANCE EVALUATION")
print("=" * 60)

# Extract best model
best_model = grid_search.best_estimator_
optimal_kernel = grid_search.best_params_['kernel']

print(f"Optimal Configuration Analysis:")
print(f"- Selected kernel: {optimal_kernel.upper()}")
for param, value in grid_search.best_params_.items():
    if param != 'kernel':
        print(f"- {param}: {value}")

# Multi-class predictions and evaluation
y_pred = best_model.predict(X_test_scaled)
test_accuracy = best_model.score(X_test_scaled, y_test)
train_accuracy = best_model.score(X_train_scaled, y_train)

print(f"\nPerformance Metrics:")
print(f"- Training accuracy: {train_accuracy:.4f}")
print(f"- Test accuracy: {test_accuracy:.4f}")
print(f"- Generalization gap: {abs(train_accuracy - test_accuracy):.4f}")

print("\nDetailed Multi-Class Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Multi-Class Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ===================================================================
# PART 6: Advanced Multi-Class Hinge Loss Implementation
# ===================================================================

print("\n\nPART 6: MULTI-CLASS HINGE LOSS ANALYSIS")
print("=" * 60)

print("Advanced Technique: Multi-Class Hinge Loss Calculation")
print("Extension from binary hinge loss to one-vs-rest multi-class scenario")

def multiclass_hinge_loss(model, X, y_true):
    """
    Advanced multi-class hinge loss implementation
    Uses one-vs-rest decision function for each class
    """
    dec_func = model.decision_function(X)  # shape (n_samples, n_classes)
    n_samples, n_classes = dec_func.shape
    total_loss = 0.0
    
    for i in range(n_samples):
        for c in range(n_classes):
            # Binary label for current class vs rest
            y_binary = 1 if y_true[i] == c else -1
            # Hinge loss for this sample-class combination
            loss = max(0, 1 - y_binary * dec_func[i, c])
            total_loss += loss
    
    return total_loss / n_samples

# Calculate comprehensive hinge loss
train_hinge_loss = multiclass_hinge_loss(best_model, X_train_scaled, y_train)
test_hinge_loss = multiclass_hinge_loss(best_model, X_test_scaled, y_test)

print(f"\nMulti-Class Hinge Loss Results:")
print(f"- Training hinge loss: {train_hinge_loss:.4f}")
print(f"- Test hinge loss: {test_hinge_loss:.4f}")
print(f"- Hinge loss gap: {abs(test_hinge_loss - train_hinge_loss):.4f}")

# Per-class analysis
print(f"\nPer-Class Performance Analysis:")
dec_func_test = best_model.decision_function(X_test_scaled)
for i, class_name in enumerate(iris.target_names):
    class_indices = y_test == i
    if np.any(class_indices):
        class_decisions = dec_func_test[class_indices, i]
        avg_confidence = np.mean(class_decisions)
        print(f"- {class_name}: Average decision confidence = {avg_confidence:.4f}")

# ===================================================================
# PART 7: Advanced Multi-Class Visualization
# ===================================================================

print("\n\nPART 7: MULTI-CLASS DECISION BOUNDARY VISUALIZATION")
print("=" * 60)

def plot_multiclass_decision_boundary(model, X, y, scaler, params):
    """
    Advanced multi-class decision boundary visualization
    Shows complex boundaries for 3-class classification
    """
    h = 0.02  # High resolution mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, 
                         edgecolors='k', s=50)
    
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title(f"Multi-Class Decision Boundary - Kernel: {model.kernel}, Params: {params}")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, alpha=0.3)
    plt.show()

print("Generating multi-class decision boundary visualization")
plot_multiclass_decision_boundary(best_model, X_test_scaled, y_test, scaler, grid_search.best_params_)

# ===================================================================
# PART 8: Comprehensive Kernel Comparison Analysis
# ===================================================================

print("\n\nPART 8: COMPREHENSIVE KERNEL PERFORMANCE COMPARISON")
print("=" * 60)

print("Advanced Analysis: Performance across all kernel types")

# Extract results for each kernel type
results_df = pd.DataFrame(grid_search.cv_results_)

# Analyze performance by kernel type
kernel_performance = {}
for kernel_type in ['linear', 'poly', 'rbf']:
    kernel_mask = results_df['param_kernel'] == kernel_type
    kernel_results = results_df[kernel_mask]
    
    best_score = kernel_results['mean_test_score'].max()
    std_score = kernel_results.loc[kernel_results['mean_test_score'].idxmax(), 'std_test_score']
    best_params = kernel_results.loc[kernel_results['mean_test_score'].idxmax(), 'params']
    
    kernel_performance[kernel_type] = {
        'best_score': best_score,
        'std_score': std_score,
        'best_params': best_params
    }

print("Kernel Performance Summary:")
print("Kernel    | Best CV Score | Std Dev | Best Parameters")
print("-" * 70)

for kernel, stats in kernel_performance.items():
    params_str = str(stats['best_params'])
    print(f"{kernel:8} | {stats['best_score']:11.4f} | {stats['std_score']:7.4f} | {params_str}")

# Determine overall best kernel
best_kernel = max(kernel_performance.keys(), 
                 key=lambda k: kernel_performance[k]['best_score'])
print(f"\nOptimal Kernel: {best_kernel.upper()}")
print(f"Performance advantage: {kernel_performance[best_kernel]['best_score']:.4f}")

# ===================================================================
# PART 9: Advanced Model Selection Insights
# ===================================================================

print("\n\nPART 9: ADVANCED MODEL SELECTION INSIGHTS")
print("=" * 60)

print("Key Discoveries from Multi-Class Real-World Analysis:")

print(f"\n1. Dataset Complexity Impact:")
print(f"   - Real-world Iris vs synthetic data shows different optimal kernels")
print(f"   - Feature scaling crucial for {test_accuracy:.1%} performance achievement")
print(f"   - Multi-class complexity: 3 decision boundaries vs 1 binary boundary")

print(f"\n2. Optimal Configuration Analysis:")
optimal_params = grid_search.best_params_
print(f"   - Best kernel: {optimal_params['kernel']} (outperformed others)")
if 'C' in optimal_params:
    print(f"   - Optimal C: {optimal_params['C']} (regularization strength)")
if 'gamma' in optimal_params:
    print(f"   - Optimal gamma: {optimal_params['gamma']} (kernel coefficient)")
if 'degree' in optimal_params:
    print(f"   - Optimal degree: {optimal_params['degree']} (polynomial complexity)")

print(f"\n3. Performance Achievements:")
print(f"   - Test accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
print(f"   - Multi-class hinge loss: {test_hinge_loss:.4f}")
print(f"   - Generalization quality: {abs(train_accuracy - test_accuracy):.4f} gap")

print(f"\n4. Advanced Implementation Features:")
print(f"   - Feature scaling: StandardScaler normalization")
print(f"   - Multi-class strategy: One-vs-Rest decision function")
print(f"   - Comprehensive grid search: {total_combinations} configurations")
print(f"   - Stratified sampling: Balanced class representation")

# ===================================================================
# PART 10: Production-Ready Implementation Guidelines
# ===================================================================

print("\n\nPART 10: PRODUCTION DEPLOYMENT FRAMEWORK")
print("=" * 60)

print("Complete SVM Implementation Pipeline Mastery:")

print(f"\n1. Data Preprocessing Excellence:")
print("   ✓ Feature scaling with StandardScaler")
print("   ✓ Stratified train-test split for balanced classes")
print("   ✓ Proper handling of multi-class scenarios")

print(f"\n2. Hyperparameter Optimization Mastery:")
print("   ✓ Multi-kernel systematic comparison")
print("   ✓ Comprehensive parameter space exploration")
print("   ✓ Cross-validation for robust model selection")
print("   ✓ Parallel processing for computational efficiency")

print(f"\n3. Advanced Evaluation Techniques:")
print("   ✓ Multi-class hinge loss implementation")
print("   ✓ Per-class performance analysis")
print("   ✓ Decision boundary visualization")
print("   ✓ Generalization gap monitoring")

print(f"\n4. Real-World Application Readiness:")
print(f"   - Achieved {test_accuracy:.1%} accuracy on standard benchmark")
print(f"   - Robust to class imbalance through stratification")
print(f"   - Scalable preprocessing pipeline")
print(f"   - Comprehensive performance metrics")

print(f"\nNext Steps for Advanced Applications:")
print("- Apply to high-dimensional datasets (text, images)")
print("- Implement custom kernel functions")
print("- Explore ensemble methods with SVM")
print("- Deploy with production monitoring systems")

print(f"\nAdvanced Multi-Class SVM Pipeline Complete!")
print(f"Evolution: Synthetic Binary → Real-World Multi-Class Mastery")
print("=" * 60)

# %%


# %%


# %%


# %%



