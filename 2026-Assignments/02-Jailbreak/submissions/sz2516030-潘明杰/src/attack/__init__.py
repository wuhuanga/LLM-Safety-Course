"""
攻击模块初始化
"""

from .blackbox_pair import PairAttack
from .whitebox_gcg import GCGAttack

__all__ = ['PairAttack', 'GCGAttack']
