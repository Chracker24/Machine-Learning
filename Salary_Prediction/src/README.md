# Functions.py
contains all the functions that are imported and used in [Machine1.py](https://github.com/Chracker24/Machine-Learning/blob/main/Salary_Prediction/src/Machine1.py)<br>
## Functions used
1. **zscorenormalization** : *To Normalize the data using mean and standard deviation*
2. **compute_cost** : *computes the cost for each iteration*
3. **compute_gradient** : *computes the gradient at which the "w" and "b" to go*
4. **gradient_descent** : *function that updates the values of "w" and "b" and prints the Iterations*
5. **predictions** : *predicts the Salaries on the training data itself*
6. **R2** : *computes the R² value of the model using predicted and actual values*
7. **plotting** : *plots the actual and the predicted line*

Modules imported : ***NumPy, Copy, matplotlib and math***

# Machine1.py
Main parent code that trains and predicts based on the data in [Salary_Data.csv](https://github.com/Chracker24/Machine-Learning/blob/main/Salary_Prediction/Data/Salary_Data.csv)<br>
imports functions from [functions.py](https://github.com/Chracker24/Machine-Learning/blob/main/Salary_Prediction/src/functions.py)<br>
Normalization used : [**Z-Score Normalization**](https://toptipbio.com/wp-content/uploads/2018/07/Z-score-formula.jpg)<br>
Learning Rate : **0.0001**<br>
Iterations : **100000**<br>
R² score = approx **0.95**<br>

modules imported : **NumPy**, ***[functions.py](https://github.com/Chracker24/Machine-Learning/blob/main/Salary_Prediction/src/functions.py)***

# Machine1_Scikit.py
code using Sci-kit Learn library to achieve the same output which is the predict Salaries after learning from YoE vs Salaries data in [Salary_Data.csv](https://github.com/Chracker24/Machine-Learning/blob/main/Salary_Prediction/Data/Salary_Data.csv)

Highest R2 score achieved - **0.99**<br>
Lower values have also been achieved ensuring that the model works well
