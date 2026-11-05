"""Distinguish float32 batch-shape rounding from cross-origin information flow."""
import copy
import torch


def check_isolation(model, x, known):
    with torch.no_grad():
        single=model(x[:1],known[:1])
        mixed=model(x,known)[:1]
        twin=model(x[:1].repeat(2,1,1),known[:1].repeat(2,1,1))[:1]
        torch.testing.assert_close(twin,mixed,rtol=1e-6,atol=1e-6)
        precise=copy.deepcopy(model).double()
        s64=precise(x[:1].double(),known[:1].double())
        b64=precise(x.double(),known.double())[:1]
        torch.testing.assert_close(s64,b64,rtol=1e-8,atol=1e-8)
    return {'fp32_cross_batch_max_abs':float((single-mixed).abs().max()),
            'fp32_fixed_batch_intervention_max_abs':float((twin-mixed).abs().max()),
            'fp64_cross_batch_max_abs':float((s64-b64).abs().max()),'passed':True}
