import os
import json


def load_key(storage, dbname, sectionname, k):
    indexname = "{}.json".format(dbname)
    if not storage.exists(indexname):
        return None

    buf = storage.load("{}.json".format(dbname), item_type="raw")
    index_json = json.loads(buf)
    section = index_json.get(sectionname, None)
    if section is not None:
        return section.get(k, None)

    return None


def save_key(storage, dbname, sectionname, k, v):
    indexname = "{}.json".format(dbname)
    if storage.exists(indexname):
        buf = storage.load(indexname, item_type="raw")
        index_json = json.loads(buf)
        section = index_json.get(sectionname, {})
    else:
        index_json = dict()
        section = {}

    section[k] = v

    index_json[sectionname] = section
    storage.store(indexname, json.dumps(index_json), item_type="raw")


# def load_key(rootdir, dbname, sectionname, k):
#     indexname = os.path.join(rootdir, "{}.json".format(dbname))
#     if os.path.exists(indexname):
#         index_json = json.load(open(indexname, "r"))
#         section = index_json.get(sectionname, None)
#         if sectionname is not None:
#             return section.get(k, None)

#     return None


# def save_key(rootdir, dbname, sectionname, k, v):
#     indexname = os.path.join(rootdir, "{}.json".format(dbname))
#     if os.path.exists(indexname):
#         index_json = json.load(open(indexname, "r"))
#         section = index_json[sectionname]
#     else:
#         index_json = dict()
#         section = {}

#     section[k] = v

#     index_json[sectionname] = section
#     json.dump(index_json, open(indexname, "w"))


