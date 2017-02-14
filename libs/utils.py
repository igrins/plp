
def compress_list(mask, items):
    return [o for (m, o) in zip(mask, items) if m]

def flatten(l):
    return [r_ for r1 in l for r_ in r1]


