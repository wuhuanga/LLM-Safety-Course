"""src包初始化。

离线脱敏评估不需要加载transformers/torch；真实模型路径使用这些对象时，
请先安装requirements.txt中的依赖。
"""

try:
    from .model_utils import ModelManager
except ModuleNotFoundError:
    ModelManager = None

from .data_utils import HarmfulBehaviorDataset, BenignPromptsDataset

try:
    from .attack import PairAttack, GCGAttack
except ModuleNotFoundError:
    PairAttack = None
    GCGAttack = None

try:
    from .defense import PerplexityFilter, SmoothLLM, LlamaGuardFilter
except ModuleNotFoundError:
    PerplexityFilter = None
    SmoothLLM = None
    from .defense.llama_guard_filter import LlamaGuardFilter

from .evaluation import EvaluationMetrics, CaseStudyGenerator

__all__ = [
    'ModelManager',
    'HarmfulBehaviorDataset',
    'BenignPromptsDataset',
    'PairAttack',
    'GCGAttack',
    'PerplexityFilter',
    'SmoothLLM',
    'LlamaGuardFilter',
    'EvaluationMetrics',
    'CaseStudyGenerator'
]
