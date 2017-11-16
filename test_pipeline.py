from igrins.pipeline import create_pipeline
from igrins.driver import Step


def test1():
    def step1(obsset):
        print(1)

    def step2(obsset, lacosmic_thresh):
        print(2, lacosmic_thresh)

    steps = [Step("step 1", step1, args0=True),
             Step("step 2", step2,
                  lacosmic_thresh=0.)]

    f = create_pipeline("flat", steps)
    print(f.__doc__)




if __name__ == "__main__":
    test1()
