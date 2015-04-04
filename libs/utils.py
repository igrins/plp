
def compress_list(mask, items):
    return [o for (m, o) in zip(mask, items) if m]
