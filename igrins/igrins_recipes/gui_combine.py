from functools import lru_cache

from ..utils.astropy_smooth import get_smoothed
from ..utils.gui_box import AnchoredGuiBox


def setup_gui(im, vmin, vmax,
              func_get_pattern_remove_n_smoothed,
              func_save):

    ax = im.axes
    fig = ax.figure

    box = AnchoredGuiBox(fig, ax, 80, align="center", pad=0, sep=5)
    ax.add_artist(box)

    vmax_input = box.gui.append_labeled_textbox("vmax", 30, 20,
                                                initial_text=str(vmax))

    vmin_input = box.gui.append_labeled_textbox("vmin", 30, 20,
                                                initial_text=str(vmin))

    check_buttons1 = box.gui.append_check_buttons(["Smooth"],
                                                  [False])

    radio_labels = ('Guard', 'Level 2', 'Level 3')
    radio_buttons = box.gui.append_radio_buttons(radio_labels, active=1)

    check_buttons2 = box.gui.append_check_buttons(["Amp-wise"],
                                                  [False])

    # box.gui.append_label(label)

    save_button = box.gui.append_button("Save & Quit")
    save_button.on_clicked(func_save)

    def get_params():
        label = radio_buttons.value_selected
        i = radio_labels.index(label)
        remove_level = i + 1

        r = dict(remove_level=remove_level,
                 amp_wise=check_buttons2.get_status()[0])
        return r

    def hzfunc(*kl, im=im, ax=ax):
        # label = radio_buttons.value_selected
        # smooth = check_buttons1.get_status()[0]
        # amp_wise = check_buttons2.get_status()[0]
        p = get_params()
        smooth = check_buttons1.get_status()[0]

        d2 = func_get_pattern_remove_n_smoothed(p["remove_level"],
                                                p["amp_wise"], smooth)

        im.set_data(d2)

        fig.canvas.draw()

    def change_clim(*kl, im=im):
        vmin = float(vmin_input.text)
        vmax = float(vmax_input.text)

        im.set_clim(vmin, vmax)
        fig.canvas.draw()

    vmax_input.on_submit(change_clim)
    vmin_input.on_submit(change_clim)
    hzfunc()

    radio_buttons.on_clicked(hzfunc)
    check_buttons1.on_clicked(hzfunc)
    check_buttons2.on_clicked(hzfunc)

    return box, get_params


def factory_pattern_remove_n_smoothed(remove_pattern,
                                      data_minus_raw,
                                      bias_mask):

    # def get_pattern_removed(remove_level, remove_amp_wise_var,
    #                         data_minus_raw=data_minus_raw, mask=bias_mask):
    #     d2 = remove_pattern(data_minus_raw, mask=mask,
    #                         remove_level=remove_level,
    #                         remove_amp_wise_var=remove_amp_wise_var)

    #     return d2

    @lru_cache(maxsize=32)
    def get_pattern_remove_n_smoothed(remove_level,
                                      amp_wise, smooth):
        d2 = remove_pattern(data_minus_raw, mask=bias_mask,
                            remove_level=remove_level,
                            remove_amp_wise_var=amp_wise)
        # d2 = get_pattern_removed(remove_level=remove_level,
        #                          remove_amp_wise_var=amp_wise)
        if smooth:
            d2 = get_smoothed(d2)

        return d2

    return get_pattern_remove_n_smoothed


def factory_processed_n_smoothed(_process):

    @lru_cache(maxsize=16)
    def get_processed(**kw):
        d2 = _process(**kw)

        return d2

    @lru_cache(maxsize=32)
    def get_processed_n_smoothed(smooth=False, **kw):
        d2 = get_processed(**kw)
        # d2 = get_pattern_removed(remove_level=remove_level,
        #                          remove_amp_wise_var=amp_wise)
        if smooth:
            d2 = get_smoothed(d2)

        return d2

    return get_processed_n_smoothed


def setup_gui_combine_sky(im, vmin, vmax,
                          params,
                          func_process,
                          func_save):

    ax = im.axes
    fig = ax.figure

    func_get_processed_n_smoothed = factory_processed_n_smoothed(func_process)

    box = AnchoredGuiBox(fig, ax, 80, align="center", pad=0, sep=5)
    ax.add_artist(box)

    vmax_input = box.gui.append_labeled_textbox("vmax", 30, 20,
                                                initial_text=str(vmax))

    vmin_input = box.gui.append_labeled_textbox("vmin", 30, 20,
                                                initial_text=str(vmin))

    check_buttons1 = box.gui.append_check_buttons(["Smooth"],
                                                  [False])

    radio_labels1 = ('level 1', 'level 2')
    # radio_return_values1 = [1, 2]
    # a = radio_return_values1.index(params["remove_level"])
    v, vv = params["remove_level"], [1, 2]
    radio_buttons1 = box.gui.append_radio_buttons(radio_labels1,
                                                  active=vv.index(v),
                                                  values=vv)

    radio_labels2 = ('none', 'sky')
    a = radio_labels2.index(params["bg_subtraction_mode"])
    radio_buttons2 = box.gui.append_radio_buttons(radio_labels2, active=a)

    # box.gui.append_label(label)

    save_button = box.gui.append_button("Save & Quit")
    save_button.on_clicked(func_save)

    def get_params():
        # label1 = radio_buttons1.value_selected
        # i = radio_labels1.index(label1)
        # remove_level = radio_return_values1[i]
        remove_level = radio_buttons1.get_selected_value()

        bg_subtraction_mode = radio_buttons2.get_selected_value()

        r = dict(remove_level=remove_level,
                 bg_subtraction_mode=bg_subtraction_mode)
        return r

    def hzfunc(*kl, im=im, ax=ax):
        p = get_params()
        smooth = check_buttons1.get_status()[0]

        d2 = func_get_processed_n_smoothed(**p, smooth=smooth)
        # d2 = func_get_pattern_remove_n_smoothed(p["remove_level"],
        #                                         p["amp_wise"], smooth)

        im.set_data(d2)

        fig.canvas.draw()

    def change_clim(*kl, im=im):
        vmin = float(vmin_input.text)
        vmax = float(vmax_input.text)

        im.set_clim(vmin, vmax)
        fig.canvas.draw()

    vmax_input.on_submit(change_clim)
    vmin_input.on_submit(change_clim)
    # hzfunc()

    radio_buttons1.on_clicked(hzfunc)
    radio_buttons2.on_clicked(hzfunc)
    check_buttons1.on_clicked(hzfunc)
    # check_buttons2.on_clicked(hzfunc)

    return box, get_params


def main():
    import matplotlib.pyplot as plt

    import igrins
    from igrins.igrins_recipes.recipe_combine import get_combined_images
    from igrins.igrins_recipes.recipe_combine import remove_pattern

    obsdate = "20190413"
    band = "K"
    obsids = range(97, 103)
    # obsids = range(68, 72)
    recipe = "STELLAR_ONOFF"

    obsset = igrins.get_obsset(obsdate, band, recipe,
                               # obsids=range(9, 12),
                               obsids=obsids,
                               # obsids=range(12, 15),
                               # frametypes="A B A A B A".split())
                               frametypes="A B A A B A".split())

    # from igrins.igrins_recipes.recipe_combine import select_k_to_remove

    # data_minus_raw = get_combined_images(obsset)
    data_minus_raw, data_plus = get_combined_images(obsset)

    bias_mask = obsset.load_resource_for("bias_mask")

    # def get_pattern_removed(remove_level, remove_amp_wise_var,
    #                         data_minus_raw=data_minus_raw, mask=bias_mask):
    #     d2 = remove_pattern(data_minus_raw, mask=mask,
    #                         remove_level=remove_level,
    #                         remove_amp_wise_var=remove_amp_wise_var)

    #     return d2

    # @lru_cache(maxsize=32)
    # def get_pattern_remove_n_smoothed(remove_level,
    #                                   amp_wise, smooth):
    #     d2 = get_pattern_removed(remove_level=remove_level,
    #                              remove_amp_wise_var=amp_wise)
    #     if smooth:
    #         d2 = get_smoothed(d2)

    #     return d2
    get_im = factory_pattern_remove_n_smoothed(remove_pattern,
                                               data_minus_raw,
                                               bias_mask)

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    vmin, vmax = -30, 30
    # setup figure guis

    obsdate, band = obsset.get_resource_spec()
    obsid = obsset.master_obsid

    def save(*kl):
        plt.close(fig)
        print("save")
        pass

    ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

    # add callbacks
    d2 = get_im(1, False, False)
    im = ax.imshow(d2, origin="lower", interpolation="none")
    im.set_clim(vmin, vmax)

    box, get_params = setup_gui(im, vmin, vmax,
                                get_im, save)

    plt.show()
    print("Next")
    print(get_params())


def main_combine_sky():
    import matplotlib.pyplot as plt

    from igrins import get_obsset
    from igrins.igrins_recipes.recipe_combine import get_combined_images
    from igrins.igrins_recipes.recipe_combine import remove_pattern

    band = "K"
    config_file = None

    obsset = get_obsset("20190318", band, "SKY",
                        obsids=range(10, 11),
                        frametypes=["-"],
                        config_file=config_file)

    d = obsset.get_hdus()[0].data

    def _process(d=d, **kwargs):
        print(kwargs)
        return d

    get_im = factory_processed_n_smoothed(_process)

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    d = get_im()
    im = ax.imshow(d, origin="lower", interpolation="none")


    def save(*kl):
        plt.close(fig)
        print("save")
        pass

    vmin, vmax = -10, 60
    box, get_params = setup_gui_combine_sky(im, vmin, vmax,
                                            get_im, save)

    plt.show()
    print("Next")
    print(get_params())


if __name__ == '__main__':
    main_combine_sky()