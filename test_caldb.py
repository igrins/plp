from igrins.libs.resource_manager import (get_file_storage,
                                          get_resource_db,
                                          get_resource_manager)

from igrins.libs.resource_manager import IgrinsBasenameHelper as BasenameHelper


class TestStorage(object):
    data = {("OUTDATA_PATH", "test_path1"): b"test_data1",
            ("INDATA_PATH", "test_SDCH_20170410_0041"): b"test_data0041"}

    # def _get_key(self, basename, item_desc):
    #     section, tmpl = item_desc
    #     return (section, tmpl.format(basename=basename))

    def load(self, section, fn, item_type=None):
        return self.data[(section, fn)]

    def store(self, section, fn, d, item_type=None):
        self.data[(section, fn)] = d


def get_test_storage(config, resource_spec):
    return TestStorage()


def get_test_resource_manager(config, resource_spec, basename_helper=None):

    from igrins.resource_manager import (ResourceStack,
                                         ResourceStackWithBasename)

    storage = get_test_storage(config, resource_spec)

    resource_db = get_resource_db(config, resource_spec)

    if basename_helper is None:
        resource_manager = ResourceStack(resource_spec, storage,
                                         resource_db)
    else:
        resource_manager = ResourceStackWithBasename(resource_spec, storage,
                                                     resource_db,
                                                     basename_helper)

    return resource_manager


def test_resource_manager(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    resource_manager = get_test_resource_manager(config, resource_spec)

    resource_manager.new_context("1st Context")

    basename = "path1"

    item_desc1 = ("OUTDATA_PATH", "test_{basename}")
    r1 = resource_manager.load(basename, item_desc1)
    assert r1 == b"test_data1"

    item_desc2 = ("OUTDATA_PATH", "test_{basename}.fits")
    data = b"TEEST"
    resource_manager.store(basename, item_desc2, data)
    r2 = resource_manager.load(basename, item_desc2)
    assert r2 == data

    resource_manager.close_context("1st Context")

    #

    resource_manager.new_context("2nd Context")

    item_desc3 = ("OUTDATA_PATH", "test_{basename}.fits")
    r3 = resource_manager.load(basename, item_desc3)
    assert r3 == b"TEEST"

    item_desc4 = ("OUTDATA_PATH", "test_{basename}.json")
    data = b"JSON"
    resource_manager.store(basename, item_desc4, data)
    r4 = resource_manager.load(basename, item_desc4)
    assert r4 == data

    resource_manager.close_context("2nd Context")


def test_resource_db(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    resource_db = get_resource_db(config, resource_spec)

    resource_db.update_db("a0v", "TEST_0020")

    resource_basename = resource_db.query_resource_basename("a0v",
                                                            "TEST_0021")
    assert resource_basename == "TEST_0020"

    # with postfix
    postfix = "_uniform"
    resource_db.update_db("a0v", "TEST_0040", postfix)

    resource_basename = resource_db.query_resource_basename("a0v",
                                                            "TEST_0021",
                                                            postfix=postfix)
    assert resource_basename == "TEST_0040"


def test_storage(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    storage = get_file_storage(config, resource_spec)

    section = "OUTDATA_PATH"
    storage.store(section, "test.txt",
                  b"JUNK")

    d = storage.load(section, "test.txt")

    assert d == b"JUNK"


def test_storage_gzip(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    storage = get_file_storage(config, resource_spec)

    section = "OUTDATA_PATH"

    def gzip_data(data):
        import StringIO
        import gzip
        out = StringIO.StringIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as f:
            f.write(data)
        return out.getvalue()

    section = "OUTDATA_PATH"
    storage.store(section, "test_gz.txt.gz",
                  gzip_data(b"JUNK"))

    d = storage.load(section, "test_gz.txt", check_candidate=True)

    assert d == b"JUNK"

    d = storage.load(section, "test_bz2.txt", check_candidate=True)

    assert d.strip() == b"JUNK2"


def test_resource_manager_w_storage(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    resource_manager = get_resource_manager(config, resource_spec)

    resource_manager.new_context("1st Context")

    basename = "path1"

    item_desc1 = ("OUTDATA_PATH", "test_{basename}")
    r1 = resource_manager.load(basename, item_desc1)
    assert r1.strip() == b"test_data1"

    item_desc2 = ("OUTDATA_PATH", "test_{basename}.fits")
    data = b"TEEST"
    resource_manager.store(basename, item_desc2, data)
    r2 = resource_manager.load(basename, item_desc2)
    assert r2 == data

    resource_manager.close_context("1st Context")

    #

    resource_manager.new_context("2nd Context")

    item_desc3 = ("OUTDATA_PATH", "test_{basename}.fits")
    r3 = resource_manager.load(basename, item_desc3)
    assert r3 == b"TEEST"

    item_desc4 = ("OUTDATA_PATH", "test_{basename}.json")
    data = b"JSON"
    resource_manager.store(basename, item_desc4, data)
    r4 = resource_manager.load(basename, item_desc4)
    assert r4 == data

    resource_manager.close_context("2nd Context")


def test_converter(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    base_storage = get_file_storage(config, resource_spec)

    from igrins.libs.item_convert import ItemConverter
    storage = ItemConverter(base_storage)

    import astropy.io.fits as pyfits
    import numpy as np

    hdu = pyfits.PrimaryHDU(data=np.zeros((10, 10)))
    hdul = pyfits.HDUList([hdu])

    storage.store("OUTDATA_PATH", "test_zero10x10.fits", hdul,
                  item_type="fits")

    hdul = storage.load("OUTDATA_PATH", "test_zero10x10.fits",
                        item_type="fits")

    assert hdul[0].data.shape == (10, 10)


def test_resource_stack_w_basename(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    basename_helper = BasenameHelper(utdate, band)
    resource_manager = get_resource_manager(config, resource_spec,
                                            basename_helper)

    resource_manager.new_context("1st Context")

    obsid = "41"

    item_desc1 = ("INDATA_PATH", "test_{basename}")
    r1 = resource_manager.load(obsid, item_desc1)
    assert r1.strip() == b"test_data0041"

    item_desc2 = ("INDATA_PATH", "test_{basename}.fits")
    data = b"TEEST"
    obsid = "45"
    resource_manager.store(obsid, item_desc2, data)
    r2 = resource_manager.load(obsid, item_desc2)
    assert r2 == data

    resource_manager.close_context("1st Context")


def test_resource_db_w_basename(config):

    utdate = "20170410TEST"
    band = "H"

    resource_spec = (utdate, band)

    basename_helper = BasenameHelper(utdate, band)

    resource_manager = get_resource_manager(config, resource_spec,
                                            basename_helper)

    resource_manager.new_context("1st Context")

    obsid = "95"

    resource_manager.update_db("a0v", obsid)

    resource_desc = ("a0v", "{basename}.a0v")
    _ = resource_manager.query_resource_for("94",
                                            resource_desc)
    resource_obsid, item_desc = _
    assert resource_obsid == "95"
    assert item_desc == "{basename}.a0v"

    resource_manager.close_context("1st Context")


def main():
    from igrins.libs.igrins_config import IGRINSConfig
    config = IGRINSConfig()

    test_resource_manager(config)

    test_storage(config)

    test_storage_gzip(config)

    test_resource_db(config)

    test_resource_manager_w_storage(config)

    test_converter(config)

    test_resource_stack_w_basename(config)

    test_resource_db_w_basename(config)


if __name__ == '__main__':
    main()
