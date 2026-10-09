"""Frame ordering and threshold policy for streaming evaluation."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('streaming_policy',Path(__file__).resolve().parents[1]/'inference/streaming.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

class PolicyTests(unittest.TestCase):
    def test_frame_precedes_generation_and_threshold_updates_afterward(self):
        calls=[]
        class Engine:
            frame_token_interval_threshold=.5
            def __call__(self,update):
                calls.append(('generate',update,self.frame_token_interval_threshold))
                return 'question','answer'
        engine=Engine();state={'silent':0,'consecutive':9,'threshold':.5}
        result=policy.process_frame(engine,2,lambda t:calls.append(('frame',t)),state)
        self.assertEqual(calls,[('frame',2),('generate',True,.5)])
        self.assertEqual(result,('question','answer'))
        self.assertAlmostEqual(engine.frame_token_interval_threshold,.6)
    def test_threshold_reset_requires_thirty_silent_frames(self):
        state={'silent':0,'consecutive':0,'threshold':.8}
        for _ in range(29):self.assertEqual(policy.update_threshold(state,False),.8)
        self.assertEqual(policy.update_threshold(state,False),.5)

    def test_fixed_threshold_is_applied_on_first_frame_and_survives_silence(self):
        class Engine:
            frame_token_interval_threshold=.5
            def __call__(self,update):
                self.used=self.frame_token_interval_threshold
                return None,None
        engine=Engine();state=policy.threshold_state(.7)
        for i in range(60):policy.process_frame(engine,i,lambda t:None,state)
        self.assertEqual(engine.used,.7)
        self.assertEqual(engine.frame_token_interval_threshold,.7)
        self.assertEqual(policy.threshold_state(.7)['threshold'],.7)
    def test_fixed_threshold_bounds(self):
        for value in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):policy.threshold_state(value)
        self.assertEqual(policy.threshold_state()['threshold'],.5)
