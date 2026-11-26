"""Check isolated part rejection, small-gap repair and metric root normalization."""
from pathlib import Path
import copy,sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from assembly import rebuild,boxes
from planner import semantic,contradictory_observations

def part(name,center,size):
    return dict(name=name,center=center,size=size,rotation_deg=[0,0,0],primitive='box',material_role='body')


class Assembly(unittest.TestCase):
    def test_snap_small_failure(self):
        parts=[part('seat',[0,0,.54],[.5,.48,.08]),part('leg',[0,0,.22],[.07,.07,.46])]
        fixed,stat=rebuild(copy.deepcopy(parts),[.5,.48,.58])
        self.assertIsNotNone(fixed)
        self.assertEqual(len(stat['snaps']),1)
        self.assertAlmostEqual(boxes(fixed)[:,0,2].min(),0)
        self.assertEqual(stat['components'],1)

    def test_reject_far_piece(self):
        pieces=[part('root',[0,0,.30],[.50,.50,.60]),part('floating',[.80,0,1.3],[.40,.10,.30])]
        fixed,stat=rebuild(pieces,[.6,.6,1.4])
        self.assertIsNone(fixed)
        self.assertEqual(stat['status'],'unsound_recipe')

    def test_context_nouns_not_categories(self):
        self.assertEqual(semantic({'id':'robot_floor','category':'equipment',
                                   'description':'Small mobile robot with wheels near a desk'}),'equipment')
        self.assertEqual(semantic({'id':'person_1','category':'person',
                                   'description':'A person standing behind a desk'}),'person')

    def test_distinct_same_frame_observations(self):
        first={'views':[{'index':5,'bbox':[.1,.2,.3,.5]}]}
        second={'views':[{'index':5,'bbox':[.5,.2,.7,.5]}]}
        self.assertTrue(contradictory_observations(first,second))

if __name__=='__main__': unittest.main()
