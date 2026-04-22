"""Re-export `ComplianceState` from core.types for backward compatibility.

The canonical definition lives in `core.types`. This file exists so imports
like `from core.state import ComplianceState` keep working.
"""

from core.types import ComplianceState

__all__ = ["ComplianceState"]
