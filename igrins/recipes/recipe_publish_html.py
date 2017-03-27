import os


def make_html(utdate, dirname, config_file="recipe.config"):

    from igrins.libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    fn = config.get_value('RECIPE_LOG_PATH', utdate)
    from igrins.libs.recipes import load_recipe_list
    recipe_list = load_recipe_list(fn)
    #recipe_dict = make_recipe_dict(recipe_list)

    #spec_template = env.get_template('spec.html')

    sources = []
    for r, obsids, frametypes, desc in recipe_list:
        if r in ["A0V_AB", "STELLAR_AB",
                 "EXTENDED_AB",
                 "EXTENDED_ONOFF",
                 "A0V_ONOFF",
                 "STELLAR_ONOFF",
                 ]:
            s = dict(zip(["name", "obj", "grp1", "grp2", "exptime", "recipe", "obsids", "frametypes"],
                         desc))
            #s["obsids"] = s["obsids"].strip()
            s["nexp"] = len(obsids)

            from igrins.libs.path_info import get_zeropadded_groupname
            objroot = get_zeropadded_groupname(s["grp1"])

            for band in "HK":
                p = "igrins_spec_%s_%s.html" % (objroot, band)
                if os.path.exists(os.path.join(dirname, p)):
                    s["url_%s" % band] = p

            for band in "HK":
                p = "igrins_spec_%sA0V_%s.html" % (objroot, band)
                if os.path.exists(os.path.join(dirname, p)):
                    s["url_%s_A0V" % band] = p

            sources.append(s)

            # jsname = "igrins_spec_%04d_H.js" % obsids[0]
            # ss = spec_template.render(utdate=utdate, jsname=jsname)
            # open(os.path.join(dirname, s["url_H"]), "w").write(ss)

            # jsname = "igrins_spec_%04d_K.js" % obsids[0]
            # ss = spec_template.render(utdate=utdate, jsname=jsname)
            # open(os.path.join(dirname, s["url_K"]), "w").write(ss)

    return sources

#recipe_dict["A0V_ABBA"]
#recipe_dict


def publish_html(utdate, config_file="recipe.config"):
    #utdate = "20140713"
    from igrins.libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    dirname = config.get_value("HTML_PATH", utdate)

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('jinja_templates'))
    template = env.get_template('index.html')

    sources = make_html(utdate, dirname)
    from igrins.libs.json_helper import json_dump
    json_dump(dict(utdate=utdate,
                   sources=sources),
              open(os.path.join(dirname, "summary.json"), "w"))

    s = template.render(utdate=utdate, sources=sources)
    open(os.path.join(dirname, "index.html"), "w").write(s)
