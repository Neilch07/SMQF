import copy
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from factor import factor
from factors import multi_factor
from analysis import *
from alpha101 import *
from gtja191 import *

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

class factors_alpha101(multi_factor):
    def calc_all_factors(self):     
        self.set_factor(alpha101_003())
        self.set_factor(alpha101_004())
        self.set_factor(alpha101_005())
        self.set_factor(alpha101_008())
        self.set_factor(alpha101_010())
        self.set_factor(alpha101_011())
        self.set_factor(alpha101_013())
        self.set_factor(alpha101_015())
        self.set_factor(alpha101_016())
        self.set_factor(alpha101_017())
        self.set_factor(alpha101_018())
        self.set_factor(alpha101_019())
        self.set_factor(alpha101_024())
        self.set_factor(alpha101_025())
        self.set_factor(alpha101_029())
        self.set_factor(alpha101_032())
        self.set_factor(alpha101_033())
        self.set_factor(alpha101_037())
        self.set_factor(alpha101_038())
        self.set_factor(alpha101_040())
        self.set_factor(alpha101_044())
        self.set_factor(alpha101_050())
        self.set_factor(alpha101_055())
        self.set_factor(alpha101_057())
        self.set_factor(alpha101_066())
        self.set_factor(alpha101_083())
        self.set_factor(alpha101_088())

class factors_alpha191(multi_factor):
    def calc_all_factors(self):     
        self.set_factor(alpha191_003())
        self.set_factor(alpha191_008())
        self.set_factor(alpha191_012())
        self.set_factor(alpha191_014())
        self.set_factor(alpha191_015())
        self.set_factor(alpha191_016())
        self.set_factor(alpha191_018())
        self.set_factor(alpha191_019())
        self.set_factor(alpha191_020())
        self.set_factor(alpha191_022())
        self.set_factor(alpha191_024())
        self.set_factor(alpha191_025())
        self.set_factor(alpha191_026())
        self.set_factor(alpha191_028())
        self.set_factor(alpha191_029())
        self.set_factor(alpha191_032())
        self.set_factor(alpha191_034())
        self.set_factor(alpha191_036())
        self.set_factor(alpha191_039())
        self.set_factor(alpha191_041())
        self.set_factor(alpha191_042())
        self.set_factor(alpha191_045())
        self.set_factor(alpha191_046())
        self.set_factor(alpha191_054())
        self.set_factor(alpha191_055())
        self.set_factor(alpha191_057())
        self.set_factor(alpha191_061())
        self.set_factor(alpha191_062())
        self.set_factor(alpha191_063())
        self.set_factor(alpha191_064())
        self.set_factor(alpha191_065())
        self.set_factor(alpha191_066())
        self.set_factor(alpha191_070())
        self.set_factor(alpha191_071())
        self.set_factor(alpha191_074())
        self.set_factor(alpha191_079())
        self.set_factor(alpha191_081())
        self.set_factor(alpha191_083())
        self.set_factor(alpha191_087())
        self.set_factor(alpha191_088())
        self.set_factor(alpha191_090())
        self.set_factor(alpha191_092())
        self.set_factor(alpha191_094())
        self.set_factor(alpha191_095())
        self.set_factor(alpha191_097())
        self.set_factor(alpha191_099())
        self.set_factor(alpha191_100())
        self.set_factor(alpha191_102())
        self.set_factor(alpha191_106())
        self.set_factor(alpha191_109())
        self.set_factor(alpha191_113())
        self.set_factor(alpha191_121())
        self.set_factor(alpha191_122())
        self.set_factor(alpha191_124())
        self.set_factor(alpha191_132())
        self.set_factor(alpha191_134())
        self.set_factor(alpha191_135())
        self.set_factor(alpha191_137())
        self.set_factor(alpha191_140())
        self.set_factor(alpha191_142())
        self.set_factor(alpha191_150())
        self.set_factor(alpha191_156())
        self.set_factor(alpha191_157())
        self.set_factor(alpha191_158())
        self.set_factor(alpha191_163())
        self.set_factor(alpha191_176())
        self.set_factor(alpha191_179())
        self.set_factor(alpha191_184())
        self.set_factor(alpha191_188())


        

t1=time.perf_counter()
factor_property={'universe':'top75pct','benchmark': 'cs500'}
params={'ic_return_horizons': [1,5,60], 'ic_delay_periods': [1,2,5],'ir_details': True, 'save': True}
f=factors_alpha191(params=params,factor_property=factor_property)
f.run()

t2=time.perf_counter()
print('%4s%-40s%10.4f%4s' % (" ","calc_ir_details calculating time:",t2-t1,"sec"))
