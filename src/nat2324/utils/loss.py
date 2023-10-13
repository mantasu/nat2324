import numpy as np

class Loss:
    """Loss utility class for computing (inverse) loss functions.
    
    Utility class for calculating various types of loss functions. It
    provides several static methods, each corresponding to a different
    type of loss function:

        * ``mae``: Mean Absolute Error (MAE), which calculates the
          average of the absolute differences between the true and
          predicted values.
        * ``mse``: Mean Squared Error (MSE), which calculates the
          average of the squared differences between the true and
          predicted values.
        * ``rmse``: Root Mean Squared Error (RMSE), which calculates the
          square root of the MSE.
        * ``r2``: R-squared, a statistical measure that represents the
          proportion of the variance for a dependent variable that's
          explained by an independent variable or variables.
        * ``mape``: Mean Absolute Percentage Error (MAPE), which
          calculates the average of the absolute percentage differences
          between the true and predicted values.
        * ``smape``: Symmetric Mean Absolute Percentage Error (SMAPE), a
          variation of MAPE that corrects its behavior near zero.
    
    The class can be called directly to calculate the loss of a given
    true and predicted values. Example:

        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1, 2, 3])
        >>> loss = Loss("mae")
        >>> loss(y_true, y_pred)
        0.0
    
    Args:
        loss_type (str): The type of loss function to use. It must be
            one of the following: "mae", "mse", "rmse", "r2", "mape", or
            "smape".
        is_inverse (bool, optional): Whether to invert the loss value.
            Defaults to ``False``.
    """
    def __init__(self, loss_type: str, is_inverse: bool = False):
        self.loss_type = loss_type
        self.is_inverse = is_inverse

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error (MAE)

        The Mean Absolute Error (MAE) is the average of the absolute
        differences between the true and predicted values:

            MAE = (1/n) * Σ|y_true - y_pred|
        
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.
        
        Returns:
            float: The calculated MAE.
        """
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error (MSE)

        The Mean Squared Error (MSE) is the average of the squared
        differences between the true and predicted values:
        
            MSE = (1/n) * Σ(y_true - y_pred)^2
        
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.
        
        Returns:
            float: The calculated MSE.
        """
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error (RMSE)

        The Root Mean Squared Error (RMSE) is the square root of the
        average of the squared differences between the true and
        predicted values.

            RMSE = sqrt((1/n) * Σ(y_true - y_pred)^2)
        
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.
        
        Returns:
            float: The calculated RMSE.
        """
        return np.sqrt(np.mean((y_true - y_pred)**2))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (R^2)

        R-squared, also known as the coefficient of determination, is a
        statistical measure that represents the proportion of the
        variance for a dependent variable that's explained by an
        independent variable or variables:

            R^2 = 1 - Σ(y_true - y_pred)^2 / Σ(y_true - mean(y_true))^2
        
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.
        
        Returns:
            float: The calculated R^2.
        """
        return 1 - np.sum((y_true - y_pred)**2) \
               / np.sum((y_true - np.mean(y_true))**2)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error (MAPE)

        The Mean Absolute Percentage Error (MAPE) is the average of the
        absolute percentage differences between the true and predicted
        values:

            MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
        
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.
        
        Returns:
            float: The calculated MAPE.
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error (SMAPE)

        The Symmetric Mean Absolute Percentage Error (SMAPE) is a
        variation of MAPE that corrects its behavior near zero. It is
        the average of the absolute percentage differences between the
        true and predicted values, adjusted for scale.

        SMAPE = 100/n * Σ 2 * |y_true - y_pred| / (|y_true| + |y_pred|)
         
        Args:
            y_true (numpy.ndarray): The ground truth values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The calculated SMAPE.
        """
        return 100 / len(y_true) * np.sum(2 * \
            np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Get the corresponding loss function (based on type)
        loss = getattr(self, self.loss_type)(y_true, y_pred)
        
        if self.is_inverse:
            # Invert if needed
            loss = 1 / (1 + loss)
        
        return loss
