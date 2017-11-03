from igrins.resource_manager import ResourceManager

from igrins.libs.resource_manager import (get_file_storage,
                                          get_resource_db,
                                          get_resource_manager)


class TestStorage(object):
    data = {("OUTDATA_PATH", "test_path1"): b"test_data1"}

    # def _get_key(self, basename, item_desc):
    #     section, tmpl = item_desc
    #     return (section, tmpl.format(basename=basename))

    def load(self, section, fn, item_type=None):
        return self.data[(section, fn)]

    def store(self, section, fn, d, item_type=None):
        self.data[(section, fn)] = d


def get_test_storage(config, resource_spec):
    return TestStorage()


# def get_file_storage(config, resource_spec):
#     from igrins.resource_manager.file_storage_igrins import get_storage
#     return get_storage(config, resource_spec)


# def get_resource_db(config, resource_spec):
#     from igrins.libs.storage_descriptions import load_resource_def
#     from igrins.resource_manager.resource_db_igrins \
#         import get_igrins_db_factory

#     db_factory = get_igrins_db_factory(config, resource_spec)

#     resource_def = load_resource_def()

#     return ResourceDBSet(resource_spec,
#                          db_factory, resource_def)


# def get_resource_manager(config, resource_spec):

#     storage = get_file_storage(config, resource_spec)

#     resource_db = get_resource_db(config, resource_spec)

#     resource_manager = ResourceManager(resource_spec, storage,
#                                        resource_db)

#     return resource_manager


def test_resource_manager(config):

    utdate = "TEST_20170410"
    band = "H"

    resource_spec = (utdate, band)

    storage = get_test_storage(config, resource_spec)

    resource_db = get_resource_db(config, resource_spec)

    resource_manager = ResourceManager(resource_spec, storage,
                                       resource_db)

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

    utdate = "TEST_20170410"
    band = "H"

    resource_spec = (utdate, band)

    resource_db = get_resource_db(config, resource_spec)

    resource_db.update_db("a0v", "TEST_0020")

    resource_basename = resource_db.query_resource_basename("a0v",
                                                            "TEST_0021")
    assert resource_basename == "TEST_0020"


def test_storage(config):

    utdate = "TEST_20170410"
    band = "H"

    resource_spec = (utdate, band)

    storage = get_file_storage(config, resource_spec)

    section = "OUTDATA_PATH"
    storage.store(section, "test.txt",
                  b"JUNK")

    d = storage.load(section, "test.txt")

    assert d == b"JUNK"


def test_resource_manager_w_storage(config):

    utdate = "TEST_20170410"
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

    utdate = "TEST_20170410"
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


def main():
    from igrins.libs.igrins_config import IGRINSConfig
    config = IGRINSConfig()

    test_resource_manager(config)

    test_storage(config)

    test_resource_db(config)

    test_resource_manager_w_storage(config)

    test_converter(config)


if __name__ == '__main__':
    main()
