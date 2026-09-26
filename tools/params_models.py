# Pydantic BaseModel pentru fiecare tool
from pydantic import BaseModel, Field

class CalculatorParams(BaseModel):
    expression: str = Field(
        description="The mathematical expression to evaluate (e.g., '2 + 3 * 4')",
        min_length=1
    )

class WebSearchParams(BaseModel):
    query: str = Field(
        description=" The search terms for web search",
        min_length=2
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1, le=20
    )

class GetDatetimeParams(BaseModel):
    pass  # There are no parameters for this tool, but we define a class for consistency.