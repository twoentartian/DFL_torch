import torch
from typing import Union, Optional


class Point:
    """
    A class representing an n-dimensional point using PyTorch tensors.

    Attributes:
        location (torch.Tensor): An n-dimensional tensor representing the point's location.
        N (int): The dimensionality of the point.
    """

    def __init__(self, location: Optional[torch.Tensor] = None, D: Optional[int] = None):
        """
        Initialize a Point with a location tensor or dimensionality.

        Args:
            location (torch.Tensor, optional): An n-dimensional tensor.
            N (int, optional): The dimensionality of the point. If provided without location,
                               creates a zero tensor of dimension N.
        """
        if location is not None:
            if not isinstance(location, torch.Tensor):
                raise TypeError("location must be a torch.Tensor")
            self.location = location
            self.D = location.shape[0]
        elif D is not None:
            self.D = D
            self.location = torch.zeros(D)
        else:
            raise ValueError("Either location or N must be provided")

    @classmethod
    def random(cls, D: int, norm: float = 1.0) -> 'Point':
        """
        Create a Point with randomly initialized elements, normalized to a specified norm.

        Args:
            N (int): The dimensionality of the point.
            norm (float): The norm to rescale to. Default is 1.0.

        Returns:
            Point: A new Point with random normalized location.
        """
        location = torch.randn(D)
        current_norm = torch.norm(location, p=2)
        if current_norm > 0:
            location = location / current_norm * norm
        return cls(location=location, D=D)

    def normalize(self, norm: float = 2.0) -> torch.Tensor:
        """
        Normalize the location tensor to a specified norm.

        Args:
            norm (float): The norm to normalize to. Default is 2.0 (L2 norm).

        Returns:
            torch.Tensor: A new normalized tensor.
        """
        current_norm = torch.norm(self.location, p=norm)
        if current_norm == 0:
            return self.location.clone()
        return self.location / current_norm

    def distance_l2(self, other: 'Point') -> torch.Tensor:
        """
        Calculate the L2 (Euclidean) distance to another point.

        Args:
            other (Point): Another Point object.

        Returns:
            torch.Tensor: The L2 distance (scalar tensor).
        """
        if not isinstance(other, Point):
            raise TypeError("other must be a Point instance")
        return torch.norm(self.location - other.location, p=2)

    def distance_l1(self, other: 'Point') -> torch.Tensor:
        """
        Calculate the L1 (Manhattan) distance to another point.

        Args:
            other (Point): Another Point object.

        Returns:
            torch.Tensor: The L1 distance (scalar tensor).
        """
        if not isinstance(other, Point):
            raise TypeError("other must be a Point instance")
        return torch.norm(self.location - other.location, p=1)

    def angle(self) -> torch.Tensor:
        """
        Get the angle tensor by normalizing to unit norm (L2).
        Equivalent to normalize(1).

        Returns:
            torch.Tensor: The normalized direction tensor.
        """
        return self.normalize(1)

    # Operator overloading
    def __add__(self, other: Union['Point', torch.Tensor, float]) -> 'Point':
        """Add two points or add a scalar/tensor to a point."""
        if isinstance(other, Point):
            return Point(self.location + other.location)
        elif isinstance(other, (torch.Tensor, int, float)):
            return Point(self.location + other)
        else:
            return NotImplemented

    def __sub__(self, other: Union['Point', torch.Tensor, float]) -> 'Point':
        """Subtract two points or subtract a scalar/tensor from a point."""
        if isinstance(other, Point):
            return Point(self.location - other.location)
        elif isinstance(other, (torch.Tensor, int, float)):
            return Point(self.location - other)
        else:
            return NotImplemented

    def __mul__(self, other: Union[torch.Tensor, float]) -> 'Point':
        """Multiply point by a scalar or tensor."""
        if isinstance(other, (torch.Tensor, int, float)):
            return Point(self.location * other)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[torch.Tensor, float]) -> 'Point':
        """Right multiplication (scalar * point)."""
        return self.__mul__(other)

    def __truediv__(self, other: Union[torch.Tensor, float]) -> 'Point':
        """Divide point by a scalar or tensor."""
        if isinstance(other, (torch.Tensor, int, float)):
            return Point(self.location / other)
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Check equality between two points."""
        if not isinstance(other, Point):
            return False
        return torch.allclose(self.location, other.location)

    def __repr__(self) -> str:
        """String representation of the Point."""
        if self.D <= 10:
            return f"Point(N={self.D}, location={self.location})"
        else:
            return f"Point(N={self.D}, location={self.location[:10]}.......)"


# Example usage
if __name__ == "__main__":
    # Create two 3D points
    p1 = Point(torch.tensor([3.0, 4.0, 0.0]))
    p2 = Point(torch.tensor([0.0, 0.0, 0.0]))

    # Create a point using dimensionality
    p_empty = Point(D=100)
    print(f"Empty 100D point: {p_empty}")

    # Create random normalized points
    p_random = Point.random(D=3, norm=1.0)
    print(f"\nRandom 3D point (norm=1): {p_random}")
    print(f"Norm of random point: {torch.norm(p_random.location)}")

    p_random2 = Point.random(D=10, norm=5.0)
    print(f"\nRandom 10D point (norm=5): {p_random2}")
    print(f"Norm of random point: {torch.norm(p_random2.location)}")

    print(f"p1: {p1}")
    print(f"p2: {p2}")

    # Normalize
    print(f"\np1 normalized (L2): {p1.normalize()}")
    print(f"p1 normalized (L1): {p1.normalize(1)}")

    # Distances
    print(f"\nL2 distance: {p1.distance_l2(p2)}")
    print(f"L1 distance: {p1.distance_l1(p2)}")

    # Angle
    print(f"\nAngle tensor: {p1.angle()}")

    # Operations
    p3 = p1 + p2
    print(f"\np1 + p2: {p3}")

    p4 = p1 - p2
    print(f"p1 - p2: {p4}")

    p5 = p1 * 2
    print(f"p1 * 2: {p5}")

    p6 = p1 / 2
    print(f"p1 / 2: {p6}")

    print(f"\np1 == p2: {p1 == p2}")
    print(f"p1 == p1: {p1 == p1}")