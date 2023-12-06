from enum import Enum
import math
from typing import Dict, List, Union


class LocalImpactCalculationMethod(Enum):
    """
    Enumeration for different methods of calculating the impact of an ablation in GNN.

    Attributes:
        ABSOLUTE_DIFFERENCE: Uses absolute difference between predictions before and after ablation.
        PROBABILITY_CHANGE: Measures the change in probability distributions (useful in classification tasks).
        CLASS_DIFFERENCE_MATRIX: Utilizes a user-defined matrix to quantify the impact of changing from one class to another.
    """

    ABSOLUTE_DIFFERENCE = 1
    PROBABILITY_CHANGE = 2
    CLASS_DIFFERENCE_MATRIX = 3


class LocalImpactCalculator:
    """
    A class to calculate the impact of changes in GNN predictions.

    Attributes:
        method (LocalImpactCalculationMethod): The method used for calculating impact.
        class_difference_matrix (dict, optional): A matrix defining the impact of changing from one class to another.

    Methods:
        calculate_impact(prev_prediction, current_prediction): Calculates the impact based on the specified method.
    """

    def __init__(
        self,
        method: LocalImpactCalculationMethod,
        class_difference_matrix: Union[
            Dict[str, Dict[str, Union[float, int]]], None
        ] = None,
    ):
        """
        Initializes the ImpactCalculator with a specific impact calculation method.

        Args:
            method (ImpactCalculationMethod): The method to use for impact calculation.
            class_difference_matrix (dict, optional): Class difference matrix for CLASS_DIFFERENCE_MATRIX method.
        """
        if (
            class_difference_matrix is None
            and method == LocalImpactCalculationMethod.CLASS_DIFFERENCE_MATRIX
        ):
            raise ValueError(
                "No class difference matrix was provided when initializing the ImpactCalculator, but the CLASS_DIFFERENCE_MATRIX method was specified."
            )

        self.method: LocalImpactCalculationMethod = method
        self.class_difference_matrix = class_difference_matrix

    def calculate_impact(
        self,
        prev_prediction: Union[float, List[float], str],
        current_prediction: Union[float, List[float], str],
    ):
        """
        Calculates the impact based on the change in predictions.

        Args:
            prev_prediction: The GNN prediction before ablation.
            current_prediction: The GNN prediction after ablation.

        Returns:
            float: The calculated impact.
        """
        if (
            self.method == LocalImpactCalculationMethod.ABSOLUTE_DIFFERENCE
            and isinstance(prev_prediction, float)
            and isinstance(current_prediction, float)
        ):
            return self._absolute_difference(prev_prediction, current_prediction)
        elif (
            self.method == LocalImpactCalculationMethod.PROBABILITY_CHANGE
            and isinstance(prev_prediction, list)
            and isinstance(current_prediction, list)
        ):
            return self._probability_change(prev_prediction, current_prediction)
        elif (
            self.method == LocalImpactCalculationMethod.CLASS_DIFFERENCE_MATRIX
            and isinstance(prev_prediction, str)
            and isinstance(current_prediction, str)
        ):
            return self._class_difference_impact(prev_prediction, current_prediction)
        else:
            raise ValueError(
                "No impact calculation method specified was provided when initializing the ImpactCalculator or the method provided is not supported."
            )

    def _absolute_difference(self, prev: float, current: float) -> float:
        """
        Calculates the absolute difference between two numerical predictions.

        Args:
            prev (float): The prediction before ablation.
            current (float): The prediction after ablation.

        Returns:
            float: The absolute difference between the two predictions.
        """
        return abs(current - prev)

    def _probability_change(self, prev: List[float], current: List[float]) -> float:
        """
        Calculates the change in probability distributions.

        Args:
            prev (List[float]): The probability distribution before ablation.
            current (List[float]): The probability distribution after ablation.

        Returns:
            float: A measure of the change in the probability distributions.
        """
        # Ensure prev and current have the same length
        if len(prev) != len(current):
            raise ValueError(
                "prev and current probability lists must be of the same length."
            )

        return math.sqrt(sum((p - c) ** 2 for p, c in zip(prev, current)))

    def _class_difference_impact(self, prev: str, current: str) -> float:
        """
        Uses the class difference matrix to calculate the impact of a class change.

        Args:
            prev (str): The class label before ablation.
            current (str): The class label after ablation.

        Returns:
            float: The calculated impact based on the class difference matrix.
        """
        if not self.class_difference_matrix:
            raise ValueError(
                "No class difference matrix was provided when initializing the ImpactCalculator."
            )
        if prev not in self.class_difference_matrix:
            raise ValueError(f"Class {prev} is not in the class difference matrix.")

        return self.class_difference_matrix[prev].get(current, 0)
