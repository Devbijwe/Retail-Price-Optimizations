import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def retail():
    data = pd.read_csv("retail_price.csv")
    data["comp_price_diff"] = data["unit_price"] - data["comp_1"]

    x = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
    y = data["total_price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42
                                                        )
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
   

    return    y_pred ,mse

if __name__ == "__main__":
    pred_output,mse=retail()
    print("Predicted: ",pred_output)
    print(f"Mean Squared Error (MSE): {mse}")
