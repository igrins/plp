from ..pipeline.steps import Step
from ..procedures.procedures_flat import (obsset_combine_flat_off,
                                          make_hotpix_mask,
                                          combine_flat_on,
                                          make_deadpix_mask,
                                          identify_order_boundaries,
                                          trace_order_boundaries,
                                          stitch_up_traces,
                                          make_bias_mask,
                                          update_db,
                                          obsset_combine_flat_off_step2)


def obsset_produce_qa_plots(obsset):
    from ..qa.flat_qa import produce_qa
    produce_qa(obsset)


steps = [Step("Combine Flat-Off", obsset_combine_flat_off,
              flat_off_pattern_removal="guard"), # guard' | 'none'
         Step("Hotpix Mask", make_hotpix_mask,
              sigma_clip1=100, sigma_clip2=5),
         Step("Combine Flat-On", combine_flat_on,
              flat_on_pattern_removal="guard"), #  # guard' | 'none'
         Step("Deadpix Mask", make_deadpix_mask,
              deadpix_thresh=0.6, smooth_size=9),
         Step("Identify Order Boundary", identify_order_boundaries),
         Step("Trace Order Boundary", trace_order_boundaries),
         Step("Stitch Up Traces", stitch_up_traces),
         Step("Bias Mask", make_bias_mask),
         Step("Update DB", update_db),
         Step("Combine Flat-Off (2nd phase)", obsset_combine_flat_off_step2),
         Step("Produce QA plots", obsset_produce_qa_plots)]
