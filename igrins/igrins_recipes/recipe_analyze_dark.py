from ..pipeline.steps import Step
from ..procedures.procedure_dark import (make_guard_n_bg_subtracted_images,
                                         estimate_amp_wise_noise,
                                         print_out_stat_summary,
                                         analyze_amp_wise_fft,
                                         analyze_c64_wise_fft,
                                         store_qa)


steps = [Step("Make RO pattern-subtracted images",
              make_guard_n_bg_subtracted_images, use_bias_mask=True),
         Step("Esimate amp-wise noise",
              estimate_amp_wise_noise),
         Step("Printing out the stat summary",
              print_out_stat_summary),
         Step("FFT analysis of RO noise: amp-wise",
              analyze_amp_wise_fft),
         Step("FFT analysis of RO noise: c64-wise",
              analyze_c64_wise_fft),
         Step("Producing QA plots",
              store_qa, qa_outtype="png")
]

