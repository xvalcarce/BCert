import unittest
from example.bcert_simple import certification as simple_cert
from example.bcert_parallelized import certification as par_cert
from example.bcert_user_parallelized import certification as usr_par_cert

class TestSingleThread(unittest.TestCase):
    def test_bc_single_thread(self):
        try:
            simple_cert()
        except:
            self.fail("Fail to run boundCert (single thread)")

class TestMultiThread(unittest.TestCase):
    def test_bc_par(self):
        try:
            par_cert()
        except:
            self.fail("Fail to run boundCertPar (mutlithread-thread, alg generated.)")

    def test_bc_user_par(self):
        try:
            usr_par_cert()
        except:
            self.fail("Fail to run boundCertUserPar (mutlithread-thread, usr generated.)")

if __name__ == '__main__':
    unittest.main()
