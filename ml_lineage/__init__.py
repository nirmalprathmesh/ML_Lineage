"""
ML Lineage Studio — Core Library
================================
Provides reusable, project-aware MLOps tracking, provenance, and verification services.
"""

from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5

__version__ = "0.1.0"
__all__ = ["ProjectContext", "compute_file_md5", "__version__"]
