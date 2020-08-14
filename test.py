import unittest
from example.bcert_simple import certification as simple_cert
from example.bcert_parallelized import certification as par_cert
from example.bcert_user_parallelized import certification as usr_par_cert
from example.maximization_single import find_maximum as find_max

class TestSingleThread(unittest.TestCase):
    def test_bc_single_thread(self):
        try:
            print("\n")
            simple_cert()
        except:
            self.fail("Fail to run boundCert (single thread)")

class TestMultiThread(unittest.TestCase):
    def test_bc_par(self):
        try:
            print("\n")
            par_cert()
        except:
            self.fail("Fail to run boundCertPar (mutlithread-thread, alg generated.)")

    def test_bc_user_par(self):
        try:
            print("\n")
            usr_par_cert()
        except:
            self.fail("Fail to run boundCertUserPar (mutlithread-thread, usr generated.)")

class TestOptimization(unittest.TestCase):
    def test_find_max(self):
        try:
            print("\n")
            find_max()
        except:
            self.fail("Maximization on single thread failed.")

if __name__ == '__main__':
    unittest.main()
