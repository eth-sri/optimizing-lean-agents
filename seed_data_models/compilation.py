"""
Compilation result data model.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompilationResult:
    """Represents the compilation status of proof code."""
    passed: bool
    complete: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    system_errors: Optional[str] = None

    def is_successful(self) -> bool:
        """True if code passed compilation and has no sorry statements."""
        return self.passed and self.complete
