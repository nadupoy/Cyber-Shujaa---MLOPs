# 🕸️ Cyber Shujaa - MLOPs 🚀

This is my solution to the week 9 assignment in the Cyber Shujaa program 
for the **Data and AI Specialist** track.

## 🧭 Table of contents

- [🌟 Overview](#🌟-overview)
  - [The assignment 🎯](#the-assignment-🎯)
  - [Links 🔗](#links-🔗)
- [🛠️ My process](#🛠️-my-process)
  - [Built with 🧱](#built-with-🧱)
  - [What I learned 🧠](#what-i-learned-🧠)
  - [Continued development 🌱](#continued-development-🌱)
  - [Useful resources 📚](#useful-resources-📚)
- [👩‍💻 Author](#👩‍💻-author)

## 🌟 Overview

### The assignment 🎯

In this assignment, we were to build an end-to-end machine learning workflow to 
predict housing prices using the **California Housing dataset**. The focus was on 
applying key concepts covered in class: **preprocessing**, **cross-validation**, 
**hyperparameter tuning**, **pipeline construction**, and **model deployment**.

The **California Housing dataset** is a well-known regression dataset that contains 
information collected from the 1990 California census. It is available directly 
through `scikit-learn`.

Target Variable:

- Median house value (in $100,000s)

Features (8):
- **MedInc**: median income in block group
- **HouseAge**: median house age in block group
- **AveRooms**: average number of rooms per household
- **AveBedrms**: average number of bedrooms per household
- **Population**: block group population
- **AveOccup**: average number of household members
- **Latitude**: block group latitude
- **Longitude**: block group longitude

The following tasks were to be completed:
1. Load the dataset using `fetch_california_housing`
2. Apply preprocessing using `ColumnTransformer`
3. Split the data into training and test sets (80/20 split)
4. Define an evaluation metric: R² Score
5. Use a `KNeighborsRegressor`
6. Apply `GridSearchCV` to tune the following hyperparameters: `n_neighbors`: 
[3, 5, 7, 9] weights: ['uniform', 'distance'] p: [1, 2]
7. Use 5-fold cross-validation
8. Save the trained pipeline using pickle or joblib

(OPTIONAL)

9. Create a simple Flask or FastAPI app with a /predict endpoint
10. Load the .pkl model and return predictions for incoming JSON input


### Links 🔗

- [Google Colab]() assignment submission

## 🛠️ My process

### Built with 🧱

- **[Python](https://www.python.org)** 🐍
- **[Pandas](https://pandas.pydata.org/docs/index.html)** Python library used for 
working with data sets with functions for cleaning, exploring and manipulating data. 🐼
- **[Matplotlib](https://matplotlib.org/stable/)**, a comprehensive Python library 
for creating static, animated and interactive visualizations.
- **[`scikit-learn`](https://scikit-learn.org/stable/index.html)**, for machine 
learning and predictive data analysis in Python.

### What I learned 🧠



### Continued development 🌱

### Useful resources 📚

## 👩‍💻 Author

- LinkedIn - [Grace Sampao](https://www.linkedin.com/in/grace-sampao)
- GitHub - [@nadupoy](https://github.com/nadupoy)
- X (formerly Twitter) - [@grace_sampao](https://x.com/grace_sampao)
- [Blog](https://nadupoy.github.io/)