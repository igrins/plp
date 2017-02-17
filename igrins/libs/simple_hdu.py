class SimpleHDU(object):
    def __init__(self, header, data):
        self.header = header
        self.data = data



class ProxyHeader(object):
    def __init__(self, hdu_header, aux_info):
        self.hdu_header = hdu_header
        self.aux_info = aux_info

    def __getitem__(self, k):
        if k in self.aux_info:
            return self.aux_info[k]
        else:
            return self.hdu_header[k]

    def __setitem__(self, k, v):
        self.aux_info[k] = v


class ProxyHDU(object):
    def __init__(self, hdu, aux_info=None):
        self.hdu = hdu

        if aux_info is None:
            aux_info = dict()

        self.header = ProxyHeader(hdu.header, aux_info)

    @property
    def data(self):
        return self.hdu.data

