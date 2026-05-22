"""core — shared utility modules for the Cognitive Outsourcing benchmark.

Re-exports the most commonly used symbols so that downstream scripts can
``from core import MeaningCompiler, ToolRegistry, …`` instead of
importing each submodule individually.
"""

try:
    from .compiler import MeaningCompiler
except ImportError:
    MeaningCompiler = None

try:
    from .injection import InjectionEngine
except ImportError:
    InjectionEngine = None
from .tools import ToolRegistry
from .gpu import GPUMonitor
from .text_utils import normalize_city, CITY_ALIASES
from .prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DEV,
    TOOL_DESCRIPTIONS_TRAVEL,
    TOOL_DESCRIPTIONS_DEV,
    TEACHER_PLANNING_PROMPT,
    TEACHER_CONVERSATION_PROMPT,
    RECALL_SYSTEM_PROMPT,
    SIG_ANSWER_REMINDER,
    LOCAL_CO_PROMPT,
    NODE_PATTERN,
)
from .metrics import init_metrics, extract_key_facts, evaluate_answer_quality, average_metrics
from .info_theory import (
    kl_divergence,
    js_divergence,
    shannon_entropy,
    shannon_entropy_array,
    mutual_information_estimate,
    mutual_information_text,
    cosine_similarity,
    head_agreement_rate,
    compute_layer_shifts,
)
