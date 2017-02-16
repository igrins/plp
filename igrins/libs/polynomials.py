import numpy.polynomial as P
import numpy.polynomial.chebyshev as cheb


def check_string(s):
    if isinstance(s, str):
        return True
    elif isinstance(s, unicode):
        return True
    else:
        return False


def convert_to_poly(p):

    if p[0].lower().startswith("poly"):
        return P.Polynomial(p[1])

    elif p[0].lower().startswith("cheb"):
        v = cheb.Chebyshev(p[1], domain=p[2], window=p[3])
        return v

    return None


def nested_convert_to_poly(l, level=0):

    #print 1, level
    if check_string(l):
        return l

    # if level > 2:
    #     print l
    #     return None

    #print 2
    try:
        n = len(l)
    except TypeError:
        return l

    #print 3
    if n > 0 and check_string(l[0]):
        v = convert_to_poly(l)
        if v is not None:
            return v

    #print 4
    l2 = [nested_convert_to_poly(l1, level=level+1) for l1 in l]
    return l2

    # for b, d in bottom_up_solutions_:
    #     import numpy.polynomial as P
    #     assert b[0] == "poly"
    #     assert d[0] == "poly"
    #     bp = P.Polynomial(b[1])
    #     dp = P.Polynomial(d[1])
    #     bottom_up_solutions.append((bp, dp))

if __name__ == "__main__":
    print nested_convert_to_poly([["poly", [1, 2, 3]]])
    print nested_convert_to_poly([["poly", [1, 2, 3]],
                                  ["poly", [1, 2, 3]]])

    # l = ('Chebyshev([-38.58445754, -50.0196254 , -47.7578578 ,   0.62804902,  -1.06017566], [    0.,  2048.], [-1.,  1.])',
    #      array([-38.58445754, -50.0196254 , -47.7578578 ,   0.62804902,  -1.06017566]),
    #      array([    0.,  2048.]),
    #      array([-1.,  1.]))

    # print nested_convert_to_poly(l)

    l = [[u'poly',
          [-36.70831668840952,
           0.15547914347378863,
           -0.0001331686992484067,
           3.062811926225611e-08,
           -6.9614038682757935e-12]],
         [u'poly',
          [32.497849958400614,
           0.12293651610678769,
           -8.773062254619747e-05,
           5.241888065536226e-09,
           -1.8583550163003756e-12]]]

    print nested_convert_to_poly(l)
