from ..pipeline.steps import Step
from ..procedures.procedure_dark import (make_pair_subtracted_images,)


steps = [Step("Make pair-subtracted images", make_pair_subtracted_images),
]

