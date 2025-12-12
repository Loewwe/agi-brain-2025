from enum import Enum
from pydantic import BaseModel

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskViolation(BaseModel):
    rule: str
    severity: RiskLevel

print("Success")
