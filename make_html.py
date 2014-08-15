import os
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('jinja_templates'))

utdate = "20140525"
dirname = "html_%s" % utdate

if 1:
    #from libs.path_info import IGRINSPath, IGRINSLog, IGRINSFiles

    #igr_path = IGRINSPath(utdate)

    #igrins_files = IGRINSFiles(igr_path)

    from libs.recipes import load_recipe_list, make_recipe_dict
    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    #recipe_dict = make_recipe_dict(recipe_list)

    spec_template = env.get_template('spec.html')

    sources = []
    for r, obsids, frametypes, desc in recipe_list:
        if r in ["A0V_AB", "STELLAR_AB",
                 "EXTENDED_AB",
                 "EXTENDED_ONOFF",
                 ]:
            s = dict(zip(["name", "obj", "grp1", "grp2", "exptime", "recipe", "obsids", "frametypes"],
                         desc))
            #s["obsids"] = s["obsids"].strip()
            s["nexp"] = len(obsids)
            s["url_H"] = "igrins_spec_%04d_H.html" % obsids[0]
            s["url_K"] = "igrins_spec_%04d_K.html" % obsids[0]
            sources.append(s)

            jsname = "igrins_spec_%04d_H.js" % obsids[0]
            ss = spec_template.render(utdate=utdate, jsname=jsname)
            open(os.path.join(dirname, s["url_H"]), "w").write(ss)

            jsname = "igrins_spec_%04d_K.js" % obsids[0]
            ss = spec_template.render(utdate=utdate, jsname=jsname)
            open(os.path.join(dirname, s["url_K"]), "w").write(ss)

#recipe_dict["A0V_ABBA"]
#recipe_dict


template = env.get_template('index.html')

s = template.render(utdate=utdate, sources=sources)
open(os.path.join(dirname, "index.html"), "w").write(s)
