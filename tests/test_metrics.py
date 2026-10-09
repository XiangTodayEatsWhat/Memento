import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmark'))
from evaluate import temporal_metrics
from judge import parse_score,aggregate,request_hash,MODEL
class MetricTests(unittest.TestCase):
    def test_window_and_empty_predictions(self):
        gt=[{'video_uid':'v','conversation':[{'role':'user','task':'object_spatial_appear','content':'q','time':0},{'role':'assistant','task':'object_spatial_appear','content':'a','time':10}]}]
        pred=[{'video_uid':'v','conversation':[{'role':'assistant','content':'late','time':15}]}]
        m,_=temporal_metrics(gt,pred);self.assertEqual(m['time_recall']['all'],0);self.assertEqual(m['redundancy'],1)
        pred[0]['conversation']=[];m,_=temporal_metrics(gt,pred);self.assertEqual(m['time_recall']['all'],0);self.assertIsNone(m['redundancy'])
    def test_judge_validation(self):
        self.assertEqual(parse_score('{"Overall Score": 7}'),7)
        for value in ['true','0','11','"8"']:
            with self.assertRaises(ValueError):parse_score('{"Overall Score": '+value+'}')
    def test_judge_requires_complete_responses(self):
        r={'custom_id':'0-0','body':{'model':MODEL},'local_reference':{'task':'object_spatial_appear','time':10}}
        metrics={'ground_truth_counts':{'all':2,'spatial':2},'matched_counts':{'all':1,'spatial':1}}
        with self.assertRaises(ValueError):aggregate([r],{},metrics)
        response={'request_sha256':request_hash(r),'requested_model':MODEL,'returned_model':MODEL,'content':'{"Overall Score": 8}'}
        result=aggregate([r],{'0-0':response},metrics);self.assertEqual(result['score_matched']['all'],8);self.assertEqual(result['score_all_zero_for_misses']['all'],4)
if __name__=='__main__':unittest.main()
