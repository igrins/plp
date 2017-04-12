from argh_helper import argh

from process_divide_a0v import process_band

from igrins.libs.recipe_factory import new_recipe_class, new_recipe_func

_recipe_class_divide_a0v = new_recipe_class("RecipeDivideA0V",
                                            ("EXTENDED_*", "STELLAR_*"),
                                            process_band)

divide_a0v = new_recipe_func("divide_a0v",
                             _recipe_class_divide_a0v)

# FIXME: This is ugly.
divide_a0v = argh.arg('-a', '--a0v', default="GROUP2")(divide_a0v)
divide_a0v = argh.arg('--a0v-obsid', default=None, type=int)(divide_a0v)
divide_a0v = argh.arg('--basename-postfix', default=None, type=str)(divide_a0v)
divide_a0v = argh.arg('--outname-postfix', default=None, type=str)(divide_a0v)
divide_a0v = argh.arg("-g", "--groups", default=None)(divide_a0v)

__all__ = divide_a0v
