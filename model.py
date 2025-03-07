from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # Для сохранения модели


def train_and_save_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Сохраняем модель
    joblib.dump(model, "house_price_model.pkl")
    
    # Сохраняем список колонок (чтобы корректно обрабатывать новые данные)
    joblib.dump(X_train.columns.tolist(), "trained_columns.pkl")

    print("✅ Модель и список признаков сохранены!")

def evaluate_model(X_test, y_test):
    model = joblib.load("model.pkl")
    y_pred = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")

