"""Real-checkpoint parity, frozen-weight and temporal-intervention checks (offline)."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

try:
    from scripts import timestral_finetune as f
    from scripts import run_timestral_finetune as r
except ImportError:
    f = r = None

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'outputs/timestral_sizes_v1/model_manifest.json'


@unittest.skipUnless(f is not None and MANIFEST.exists(), 'Timestral environment/cached weights required')
class TimestralFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.checkpoint = json.loads(MANIFEST.read_text())['Timestral-28M']['local_path']
        cls.config = json.loads((ROOT/'configs/timestral_finetune.json').read_text())

    def data(self, batch=2):
        gen=torch.Generator().manual_seed(9)
        return (torch.randn(batch,1,168,generator=gen)*50+100,
                torch.randn(batch,8,192,generator=gen), torch.randn(batch,24,generator=gen)*50+100)

    def test_differentiable_path_matches_official_unclipped_forecast(self):
        model=f.AdaptedTimestral(self.checkpoint,'ZeroShot')
        x,c,_=self.data()
        with torch.no_grad():
            raw=model(x,c)
        official=model.forecaster.forecast(x,prediction_length=24,known_covariates=c,
                                          clip_nonnegative=False,symmetric_averaging=True).quantiles[:,0]
        torch.testing.assert_close(raw,official,rtol=0,atol=0)

    def test_lora_initialization_preserves_official_predictions_and_batch_isolation(self):
        zero=f.AdaptedTimestral(self.checkpoint,'ZeroShot')
        lora=f.AdaptedTimestral(self.checkpoint,'LoRA')
        x,c,_=self.data()
        with torch.no_grad():
            torch.testing.assert_close(zero(x,c),lora(x,c),rtol=0,atol=0)
            torch.testing.assert_close(lora(x[:1],c[:1]),lora(x,c)[:1],rtol=1e-4,atol=1e-3)

    def test_optimizer_only_changes_authorized_parameters(self):
        x,c,y=self.data(batch=1)
        for mode in ('Head','LoRA'):
            model=f.AdaptedTimestral(self.checkpoint,mode)
            frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            before=model.trainable_state()
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            loss=f.pinball(model(x,c),y,model.levels)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[n],p) for n,p in model.named_parameters() if p.requires_grad))
            for n,p in model.named_parameters():
                if n in frozen:
                    self.assertTrue(torch.equal(p,frozen[n]),n)
            state=model.trainable_state()
            reload=f.AdaptedTimestral(self.checkpoint,mode)
            reload.load_trainable(state)
            with torch.no_grad():
                torch.testing.assert_close(model(x,c),reload(x,c),rtol=0,atol=0)

    def test_future_label_intervention_cannot_change_fit(self):
        days=pd.date_range('2024-01-08',periods=4)
        data=self.data(batch=4)
        changed=(data[0],data[1],data[2].clone())
        changed[2][2:]+=999999
        states=[]
        for dataset in (data,changed):
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp);(output/'checkpoints').mkdir()
                model=r.new_model(self.checkpoint,'Head',42,self.config)
                model,audit=r.fit(model,'Head',42,days,dataset,days[2],self.config,
                                  output,io.StringIO(),stage='unit',epochs=1)
                states.append(model.trainable_state())
                self.assertEqual(audit['last_training_target'],'2024-01-09')
        for n in states[0]:
            torch.testing.assert_close(states[0][n],states[1][n],rtol=0,atol=0)

    def test_replay_and_target_loss_boundaries(self):
        days=pd.date_range('2024-01-01',periods=100)
        idx=r.training_indices(days,days[90],56)
        self.assertEqual(len(idx),56)
        self.assertEqual(idx[-1],89)
        q=torch.tensor([[[0.,1.,2.]]]);y=torch.tensor([[1.]])
        self.assertAlmostEqual(float(f.pinball(q,y,[.1,.5,.9])),.2/3,places=6)
        with self.assertRaises(ValueError):
            f.pinball(torch.zeros(1,32,21),torch.zeros(1,24),[.5]*21)

    def test_fitted_lora_isolation_distinguishes_batch_rounding(self):
        from scripts.timestral_isolation import check_isolation
        path=ROOT/'outputs/timestral_finetune_v1/checkpoints/test_LoRA_seed43_2025-12-30.pt'
        if not path.exists():
            self.skipTest('Completed regression checkpoint required')
        from scripts.foundation_models import make_input
        prices,cal,_=r.b.load_hourly(ROOT/'data/raw/shandong/prices.csv')
        samples=[make_input(prices,cal,day) for day in ('2025-07-01','2025-12-30')]
        x=torch.from_numpy(np.stack([s.target for s in samples]));known=torch.from_numpy(np.stack([s.known for s in samples]))
        model=r.new_model(self.checkpoint,'LoRA',43,self.config)
        model.load_trainable(torch.load(path,weights_only=False,map_location='cpu')['trainable_state'])
        audit=check_isolation(model,x,known)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['fp32_fixed_batch_intervention_max_abs'],0.)
        self.assertLess(audit['fp64_cross_batch_max_abs'],1e-8)

    def test_isolation_rejects_actual_cross_sample_leakage(self):
        from scripts.timestral_isolation import check_isolation
        class Leaking(torch.nn.Module):
            def forward(self,x,known):
                return x.mean().expand(len(x),24,21)
        x=torch.stack([torch.zeros(1,168),torch.ones(1,168)])
        with self.assertRaises(AssertionError):
            check_isolation(Leaking(),x,torch.zeros(2,8,192))


if __name__=='__main__':
    unittest.main()
