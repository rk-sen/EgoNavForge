"""Deterministic regression for verified, ambiguous and exterior door hypotheses."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from layout import partitions,render_architecture


def env():
    return (np.array([0.,0.]),np.array([6.,4.]),2.8,'synthetic',
            [{'type':'supported_wall_plane'}]*4,
            [dict(id='P0_01',axis=0,offset_m=3.,span_m=[0.,4.],vertical_coverage=.88,tall_span_m=3.6,points=720)])


class Enclosure(unittest.TestCase):
    def test_single_door_trace_verified(self):
        trace=np.array([[1.,1.9,1.4],[2.,1.9,1.4],[3.1,1.9,1.4],[5.,1.9,1.4]])
        doors=[dict(semantic='door',xyz=np.array([3.,1.91,0]),dimensions=np.array([.93,.16,2.14]))]
        w,r,_=partitions(env(),{'partitions':[{'plane_id':'P0_01'}]},trace,doors)
        things=[]
        rooms,ledger=render_architecture(env(),w,r,lambda *a,**kw:things.append((a,kw)))
        self.assertEqual(len(rooms),2)
        interior=[v for v in ledger if not v['exterior']]
        self.assertEqual(len(interior),1)
        aperture=interior[0]['apertures'][0]
        self.assertEqual(aperture['type'],'connected_door')
        self.assertEqual(set(aperture['room_ids']),{'Room00','Room01'})
        self.assertGreaterEqual(aperture['span_m'][1]-aperture['span_m'][0],.82)
        self.assertTrue(all('connected_door' not in [p['type'] for p in i['apertures']] for i in ledger if i['exterior']))

    def test_unverified_interior_partition_rejected(self):
        trace=np.array([[1.,1.9,1.4],[5.,1.9,1.4]])
        w,r,reason=partitions(env(),{'partitions':[{'plane_id':'P0_01'}]},trace,[])
        self.assertEqual(len(w),4)
        self.assertEqual(len(r),1)
        self.assertEqual(reason[0]['status'],'unresolved_connection')

    def test_exterior_door_gets_solid_leaf(self):
        trace=np.array([[1.,1.9,1.4],[2.,1.9,1.4]])
        door=[dict(semantic='door',xyz=np.array([.01,1.90,0]),dimensions=np.array([.90,.16,2.14]))]
        walls,rooms,_=partitions(env(),{},trace,door)
        instances=[]
        _,ledger=render_architecture(env(),walls,rooms,lambda *a,**kw:instances.append(a))
        self.assertEqual(ledger[0]['apertures'][0]['type'],'closed_boundary_door')
        self.assertTrue(any(a[0]=='Exterior_0_Leaf_0' for a in instances))


if __name__=='__main__':
    unittest.main()
