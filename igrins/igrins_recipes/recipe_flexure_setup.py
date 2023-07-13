from ..pipeline.steps import Step
# from ..procedures.procedures_flexure_correction import (test1,
#                                           test2,
#                                           test3)
from ..procedures.procedures_flexure_correction import (set_reference_frame,
											estimate_flexure,
											)

# steps = [Step("Test 1", test1),
# 		Step("Test 2", test2),
# 		Step("Test 3", test3)]

steps = [Step("Create reference frames.", set_reference_frame),
		#Step('Estimate flexure', estimate_flexure)
		]
