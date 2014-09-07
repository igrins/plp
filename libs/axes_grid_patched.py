from mpl_toolkits.axes_grid1.axes_grid import Grid as GridOld
from mpl_toolkits.axes_grid1.axes_grid import _tick_only

class Grid(GridOld):
    def set_label_mode(self, mode):
        "set label_mode"
        if mode == "all":
            for ax in self.axes_all:
                _tick_only(ax, False, False)
        elif mode == "L":
            # left-most axes
            for ax in self.axes_column[0][:-1]:
                _tick_only(ax, bottom_on=True, left_on=False)
            # lower-left axes
            ax = self.axes_column[0][-1]
            _tick_only(ax, bottom_on=False, left_on=False)

            for col in self.axes_column[1:]:
                # axes with no labels
                for ax in col[:-1]:
                    _tick_only(ax, bottom_on=True, left_on=True)

                # bottom
                if col:
                    ax = col[-1]
                    _tick_only(ax, bottom_on=False, left_on=True)

        elif mode == "1":
            for ax in self.axes_all:
                _tick_only(ax, bottom_on=True, left_on=True)

            ax = self.axes_llc
            _tick_only(ax, bottom_on=False, left_on=False)
