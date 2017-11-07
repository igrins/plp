import os
import gzip
import bz2

from six import BytesIO

from collections import OrderedDict


class NoCompress(object):
    @staticmethod
    def compress(data):
        return data

    @staticmethod
    def decompress(data):
        return data


class GzipHelper(object):
    @staticmethod
    def compress(data):
        import gzip
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as f:
            f.write(data)
        return out.getvalue()

    @staticmethod
    def decompress(data):
        import gzip
        input = BytesIO(data)
        with gzip.GzipFile(fileobj=input, mode="rb") as f:
            return f.read()


class Bzip2Helper(object):
    @staticmethod
    def compress(data):
        import bz2
        return bz2.compress(data)

    @staticmethod
    def decompress(data):
        import bz2
        return bz2.decompress(data)


candidate_generators = OrderedDict()
candidate_generators[""] = (lambda x: x, NoCompress)
candidate_generators["gzip"] = (lambda x: os.path.extsep.join([x, "gz"]),
                                GzipHelper)
candidate_generators["bzip2"] = (lambda x: os.path.extsep.join([x, "bz2"]),
                                 Bzip2Helper)
