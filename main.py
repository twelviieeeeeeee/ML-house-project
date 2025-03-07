import pandas as pd

import joblib




df = pd.read_csv("data/housing.csv")

print(df.head())

# Проверяем пропущенные значения
print(df.isnull().sum())

# Основная статистика
print(df.describe())

# Типы данных
print(df.dtypes)

# Корреляция признаков с ценой
print(df.select_dtypes(include=["number"]).corr()["price"].sort_values(ascending=False))


# Удаляем строки с пропущенными значениями
df = df.dropna()

print(df.columns)  # Выведет все колонки

# Конвертируем категориальные переменные в числа
df = pd.get_dummies(df, columns=["area"], drop_first=True)

df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)


print(df.head())  # Проверяем результат

binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})  # Преобразуем в 0 и 1


from sklearn.model_selection import train_test_split

X = df.drop("price", axis=1)  # Все признаки, кроме цены
y = df["price"]  # Целевая переменная

# Разделяем данные на 80% train и 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер train: {X_train.shape}, test: {X_test.shape}")

from model import train_and_save_model, evaluate_model

train_and_save_model(X_train, y_train)  # Обучаем модель
evaluate_model(X_test, y_test)  # Тестируем


def predict_new(flat_features):
    # Загружаем модель
    model = joblib.load("house_price_model.pkl")  # Имя модели должно совпадать с тем, которое ты сохраняешь

    # Определяем список всех признаков, которые были в обучении
    feature_names = [
        "area", "bedrooms", "bathrooms", "stories", "parking", 
        "mainroad", "guestroom", "basement", "hotwaterheating", 
        "airconditioning", "prefarea", "furnishingstatus"
    ]

    # Проверяем, что количество переданных данных совпадает с количеством признаков
    if len(flat_features) != len(feature_names):
        raise ValueError(f"Ожидалось {len(feature_names)} признаков, но получено {len(flat_features)}")

    # Создаем DataFrame с входными данными
    flat_features_df = pd.DataFrame([flat_features], columns=feature_names)

    # Преобразуем бинарные признаки (yes/no → 1/0)
    binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in binary_columns:
        flat_features_df[col] = flat_features_df[col].map({"yes": 1, "no": 0})

    # Преобразуем категориальные переменные в дамми-признаки
    flat_features_df = pd.get_dummies(flat_features_df, columns=["furnishingstatus"], drop_first=True)

    # Приводим данные к формату, который использовался при обучении модели
    trained_columns = joblib.load("trained_columns.pkl")  # Загружаем список признаков, использованных при обучении
    flat_features_df = flat_features_df.reindex(columns=trained_columns, fill_value=0)

    # Делаем предсказание
    price = model.predict(flat_features_df)

    print(f"🔮 Предсказанная цена: {price[0]:,.2f} руб.")


# Пример новой квартиры
new_flat = [5000, 3, 2, 2, 1, "yes", "no", "no", "yes", "yes", "no", "semi-furnished"]
predict_new(new_flat)


import matplotlib.pyplot as plt
import seaborn as sns
    

import matplotlib.pyplot as plt
import seaborn as sns

# График реальных vs предсказанных цен
def plot_results(X_test, y_test):
    model = joblib.load("house_price_model.pkl")
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Реальная цена")
    plt.ylabel("Предсказанная цена")
    plt.title("Реальные vs Предсказанные цены")
    plt.show()

plot_results(X_test, y_test)
