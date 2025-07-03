# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF, Matern, DotProduct, RationalQuadratic, ExpSineSquared, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_validate, KFold
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    train_df = train_df.dropna(subset=['price_CHF']) 

    # Median-Imputation anwenden
    imputer = KNNImputer(n_neighbors=5)
    OHE_preprocessor = ColumnTransformer(transformers=[('categorical',OneHotEncoder(handle_unknown='ignore'
                                            , sparse_output=False), ['season'])], remainder='passthrough')

    # modify/ignore the initialization of these variables
    X_train = OHE_preprocessor.fit_transform(train_df.drop(['price_CHF'],axis=1))
    y_train = train_df['price_CHF'].values
    X_test = OHE_preprocessor.transform(test_df)

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._model = None
        self._method = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=1e-5)
        self._x_train = X_train
        self._y_train = y_train
        self._model.fit(self._x_train, self._y_train)


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

def evaluate_kernels (X_train, y_train, kernels):
    kernel_names = [str(k) for k in kernels]
    result_kernels = pd.DataFrame(
        data=[[np.nan] * len(kernels)],
        columns=kernel_names,
        index=["RMSE"]
    )
    for kernel in kernels:
        print(kernel)
        result_kernel = cross_validate(
            GaussianProcessRegressor(kernel=kernel, alpha=1e-5),
            X_train, y_train,
            scoring='r2',
            cv=KFold(n_splits=10),
            return_train_score=True
        )
        result_kernels.at["RMSE", str(kernel)] = -result_kernel['test_score'].mean()
    best_kernel_name = result_kernels.loc["RMSE"].idxmin()
    print("The best kernel is:", str(best_kernel_name),"with error: ", result_kernels.at["RMSE",best_kernel_name])
    return best_kernel_name

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    model = Model()
    #Use this function for training the model
    kernels = [
        DotProduct(),  # linear kernel
        RBF(length_scale=1.0),  # squared exponential / RBF
        Matern(length_scale=1.0, nu=1.5),  # Matern kernel
        RationalQuadratic(),  # generalization of RBF

        # --- Extended combinations ---
        DotProduct() + WhiteKernel(),  # linear + noise
        RBF(length_scale=1.0) + WhiteKernel(),  # RBF + noise
        RationalQuadratic() + WhiteKernel(),  # RationalQuadratic + noise
        ConstantKernel(1.0) * RBF(),  # scaled RBF
        DotProduct() + RBF(),  # linear + RBF
        Matern(length_scale=1.0, nu=1.5) + RBF(),  # Matern + RBF
        #ExpSineSquared() + RBF(),  # periodic + RBF
        Matern(length_scale=1.0, nu=1.5) + WhiteKernel(),  # Matern + noise
    ]

    kernel_map = {str(k): k for k in kernels}

    best_kernel_str = evaluate_kernels(X_train, y_train, kernels)
    model.train(X_train=X_train, y_train=y_train, kernel=kernel_map[best_kernel_str])
    # Use this function for inferece
    y_pred = model.predict(X_test)


    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    # dt.to_csv('results11.csv', index=False)
    print("\nResults file successfully generated!")

