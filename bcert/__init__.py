# coding: utf-8

# Classes CompactSpace and FunLipschitz are the two required classe to use bcert
from .compact_space import CompactSpace
from .lipschitz import FunLipschitz

# Main function containing the ecursive algorithm for the bound certification
from .bcert import boundCert, boundCertPar, boundCertUserPar

# Optimization of Lipschitz function on compact space
from .bcert import maxCert, maxCertPar

__version__ = "0.0.4"
