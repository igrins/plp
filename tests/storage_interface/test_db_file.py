import igrins.storage_interface.db_file as db
import json

def _get_storage(rootdir):
    from igrins.resource_manager.file_storage import FileStorage
    path_info = dict(OUTDATA=rootdir)
    storage = FileStorage(resource_spec=None, path_info=path_info)
    return storage.new_sectioned_storage("OUTDATA")


def test_load(tmpdir):
    storage = _get_storage(tmpdir.dirname)
    d = dict(qlook=dict(k="test content"))
    storage.store("index.json", json.dumps(d), item_type="raw")

    v = db.load_key(storage, dbname="index",
                    sectionname="qlook", k="k")
    assert v == "test content"


def test_save(tmpdir):
    storage = _get_storage(tmpdir.dirname)
    db.save_key(storage, dbname="index",
                sectionname="qlook", k="k1", v="test content")
    d = json.loads(storage.load("index.json", item_type="raw"))
    assert d["qlook"]["k1"] == "test content"


def test_save2(tmpdir):
    storage = _get_storage(tmpdir.dirname)
    db.save_key(storage, dbname="index",
                sectionname="qlook", k="k1", v="test content")
    db.save_key(storage, dbname="index",
                sectionname="qlook", k="k2", v="test content2")

    d = json.loads(storage.load("index.json", item_type="raw"))
    assert d["qlook"]["k1"] == "test content"
    assert d["qlook"]["k2"] == "test content2"

def test_load_save(tmpdir):
    storage = _get_storage(tmpdir.dirname)
    db.save_key(storage, dbname="index",
                sectionname="qlook", k="k1", v="test content")
    db.save_key(storage, dbname="index",
                sectionname="qlook", k="k2", v="test content2")


    v1 = db.load_key(storage, dbname="index",
                     sectionname="qlook", k="k1")
    v2 = db.load_key(storage, dbname="index",
                     sectionname="qlook", k="k2")

    assert v1 == "test content"
    assert v2 == "test content2"
