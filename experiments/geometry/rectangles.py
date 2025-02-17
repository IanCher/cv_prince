"""Classes and functions to manipulate rectangles"""

from dataclasses import dataclass
from typing import Self


@dataclass
class Rectangle:
    top_left_x: int | float
    top_left_y: int | float
    bot_right_x: int | float
    bot_right_y: int | float

    @property
    def width(self):
        return self.bot_right_x - self.top_left_x

    @property
    def height(self):
        return self.bot_right_y - self.top_left_y

    @property
    def area(self) -> int | float:
        return self.height * self.width

    def overlap_ratio(self, other: Self) -> float:
        overlap_top_left_x = max(self.top_left_x, other.top_left_x)
        overlap_bot_right_x = min(self.bot_right_x, other.bot_right_x)

        if overlap_top_left_x >= overlap_bot_right_x:
            return 0

        overlap_top_left_y = max(self.top_left_y, other.top_left_y)
        overlap_bot_right_y = min(self.bot_right_y, other.bot_right_y)

        if overlap_top_left_y >= overlap_bot_right_y:
            return 0

        overlap = Rectangle(
            top_left_x=overlap_top_left_x,
            top_left_y=overlap_top_left_y,
            bot_right_x=overlap_bot_right_x,
            bot_right_y=overlap_bot_right_y,
        )

        return overlap.area / self.area
