from sklearn.ensemble import RandomForestRegressor


class NetworkStateRandomForest:
    """
    Baseline model for predicting the next network state.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def train(self, X, y):
        """
        Train the model.
        """

        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the next network state.
        """

        return self.model.predict(X)