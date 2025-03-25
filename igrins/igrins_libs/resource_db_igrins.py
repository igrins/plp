import os

from .storage_descriptions import DB_Specs
from .path_info import IGRINSPath, ensure_dir


def get_igrins_db_factory(config, resource_spec):
    utdate, band = resource_spec
    path_info = IGRINSPath(config, utdate, band)
    return DBFactory(resource_spec, db_specs=DB_Specs,
                     path_info=path_info)


class ResourceDBBase(object):
    def __init__(self, db_desc, db_kind):
        self.db_desc = db_desc
        self.db_kind = db_kind

    def update(self, basename):
        pass

    def query(self, basename):
        pass


# IGRINS specific

class DBFactory(object):
    def __init__(self, resource_spec, db_specs, path_info):
        self.utdate, self.band = resource_spec
        self.path_info = path_info
        self.db_specs = db_specs

    def __call__(self, db_type):
        db_spec_path, db_spec_name = self.db_specs[db_type]

        db_dir = self.path_info.get_section_path(db_spec_path)

        db_path = os.path.join(db_dir, db_spec_name)

        db = ResourceDBFile(db_type, db_path,
                            db_kind=self.band)

        return db


class ResourceDBFile(ResourceDBBase):
    def __init__(self, db_desc, dbpath, db_kind):

        ResourceDBBase.__init__(self, db_desc, db_kind)

        self.dbpath = dbpath
        # db_spec_path, db_spec_name = DB_Specs[db_type]

        # db_path = self.helper.get_item_path((db_spec_path, db_spec_name),
        #                                     basename=None)
        # self.dbname = dbname

    def update(self, basename, postfix=""):
        if os.path.exists(self.dbpath):
            mode = "a"
        else:
            dirname = os.path.dirname(self.dbpath)

            ensure_dir(dirname)

            mode = "w"

        with open(self.dbpath, mode) as myfile:
            if postfix:
                myfile.write("%s %s %s\n" % (self.db_kind, basename, postfix))
            else:
                myfile.write("%s %s\n" % (self.db_kind, basename))

    def get_obsid_list(self, postfix=""):
        import os

        if not os.path.exists(self.dbpath):
            raise RuntimeError("db not yet created: %s" % self.dbpath)

        with open(self.dbpath, "r") as myfile:
            obsid_list = []
            basename_list = []
            for l0 in myfile.readlines():
                b_l1 = l0.strip().split()
                if len(b_l1) == 3:
                    b, l1, pf = b_l1
                elif len(b_l1) == 2:
                    b, l1 = b_l1
                    pf = ""
                else:
                    raise RuntimeError("")

                if (b, pf) != (self.db_kind, postfix):
                    continue

                try:
                    #obsid_ = int(l1.strip().split("_")[-1])
                    obsid_ = int(l1.strip().split("_")[0].split('S')[1])
                except ValueError:
                    continue

                obsid_list.append(obsid_)
                basename_list.append(l1.strip())

        return obsid_list, basename_list

    def query(self, basename, postfix=""):
        # import numpy as np
        # import os

        # this needs refactoring

        # if not os.path.exists(self.dbpath):
        #     raise RuntimeError("db not yet created: %s" % self.dbpath)

        # obsid_part = basename.strip().split("_")[-1]
        # obsid = int(p.split(obsid_part)[0])

        # with open(self.dbpath, "r") as myfile:
        #     obsid_list = []
        #     basename_list = []
        #     for l0 in myfile.readlines():
        #         b_l1 = l0.strip().split()
        #         if len(b_l1) == 3:
        #             b, l1, pf = b_l1
        #         elif len(b_l1) == 2:
        #             b, l1 = b_l1
        #             pf = ""
        #         else:
        #             raise RuntimeError("")

        #         if (b, pf) != (self.db_kind, postfix):
        #             continue

        #         try:
        #             obsid_ = int(l1.strip().split("_")[-1])
        #         except ValueError:
        #             continue

        #         obsid_list.append(obsid_)
        #         basename_list.append(l1.strip())

        import numpy as np

        import re
        p = re.compile(r"\D+")



        obsid_list, basename_list = self.get_obsid_list(postfix)



        #obsid_part = basename.strip().split("_")[-1].split('S')[-1]
        obsid_part = basename.strip().split("_")[0].split('S')[-1]
        obsid = int(p.split(obsid_part)[0])

        if obsid_list:
            # return last one with minimum distance
            obsid_dist = np.abs(np.array(obsid_list) - obsid)
            i = np.where(obsid_dist == np.min(obsid_dist))[0][-1]
            return basename_list[i]
        else:
            raise RuntimeError("db (%s) is empty." % (self.dbpath))
