from util import *
from evaluate import Evaluator
from scipy import fft
import matplotlib.pylab as pylab
import unittest

class TestKeyLab(unittest.TestCase):

    def test_key_at(self):
        lab_file = '13_a_day_in_the_life.lab'
        keylab = KeyLab(lab_file)
        m = simple_keymap
        self.assertEquals(m['G'], keylab.key_at(0))
        self.assertEquals(m['G'], keylab.key_at(135))
        self.assertEquals(m['E'], keylab.key_at(136))
        self.assertEquals(m['E'], keylab.key_at(150.1))
        self.assertEquals(m['G'], keylab.key_at(250.99))
        self.assertEquals(m['E'], keylab.key_at(259.227))
        self.assertEquals(None, keylab.key_at(400))

if __name__ == '__main__':
    unittest.main()
    
