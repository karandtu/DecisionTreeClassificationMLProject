import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def decision_tree_regression():
#generate synthetic data
    x,y = make_regression(n_samples=200,n_features=2,noise=10,random_state=42)

#split the data
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#perform decision tree regression using decision tree regressor
    regressor=DecisionTreeRegressor(max_depth=3,random_state=42)

#fit this model on training data
    regressor.fit(x_train,y_train)

#perform prediction on test data
    y_pred = regressor.predict(x_test)

#calculate meanSquaredError
    mse=mean_squared_error(y_test,y_pred)

    print(f"MeanSquaredError:",{mse})

#perform the plotting using specific figure size of 12 width inches 8 height inches
#plot and visualize the tree
    plt.figure(figsize=(12,8))
    plot_tree(regressor,filled=True,feature_names=["Features"],class_names=["Targets"])
    plt.title("Decision Tree Regression Plot Visualizations")
    plt.show()


if __name__=="__main__":
    decision_tree_regression()




