def get_pipeline_steps(recipe_name):
    def step1(obsset, args0=4):
        print(obsset, args0)

    def step2(obsset, lacosmic_thresh):
        print(obsset, lacosmic_thresh)

    from .driver import Step
    steps = [Step("step 1", step1, args0=True),
             Step("step 2", step2,
                  lacosmic_thresh=0.)]

    return steps
