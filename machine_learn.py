from prepare import df
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import utils
import preprocessing
import catboost as cb

y = df["target"]
X = df.drop("target", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1, random_state=1)

pool_train = cb.Pool(X_train, y_train)

model = cb.CatBoostRegressor(n_estimators=200,
                       loss_function='RMSE',
                       learning_rate=0.4,
                       depth=3, task_type='CPU',
                       random_state=1,
                       verbose=False)

model.fit(pool_train)
