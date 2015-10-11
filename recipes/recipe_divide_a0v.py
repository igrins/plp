from process_divide_a0v import process_band

from libs.recipe_factory import new_recipe_class, new_recipe_func

_recipe_class_divide_a0v = new_recipe_class("RecipeDivideA0V",
                                            ("EXTENDED_*", "STELLAR_*"),
                                            process_band)

divide_a0v = new_recipe_func("divide_a0v",
                             _recipe_class_divide_a0v)

__all__ = divide_a0v
