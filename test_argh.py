from igrins.pipeline.argh_helper import argh, arg, wrap_multi


def test_arg():
    args_drive = [arg("-b", "--bands", default="HK"),
                  arg("-g", "--groupname", default=None),
                  arg("-d", "--debug", default=False)]

    args = [arg("--lacosmic-thresh", default=2.),
            arg("--args0", default=True)]

    def flat(utdate, **kwargs):
        print(kwargs)
        # steps = get_flat_steps()
        # for s in steps:
        #     s(None)

    flat = wrap_multi(flat, args)
    flat = wrap_multi(flat, args_drive)

    import sys
    parser = argh.ArghParser()
    argh.add_commands(parser, [flat])
    print(sys.argv)
    argh.dispatch(parser, argv=sys.argv[1:])


def main():
    test_arg()


if __name__ == '__main__':
    main()
