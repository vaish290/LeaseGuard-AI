
from typing import List
from pydantic import BaseModel, Field, model_validator


class ClauseFinding(BaseModel):
    category: str = Field(
        description="General category of the lease provision"
    )

    clause_type: str = Field(
        description="Specific type of contractual clause"
    )

    triggering_condition: str = Field(
        description="Condition under which the clause applies"
    )

    obligation: str = Field(
        description="Tenant or landlord obligation"
    )

    financial_impact: str = Field(
        description="Financial consequences of the provision"
    )

    evidence_quote: str = Field(
        description="Exact quotation from a retrieved lease passage"
    )

    source_page: int = Field(
        description="Original PDF page number"
    )

    needs_clarification: bool = Field(
        description="Whether the provision requires clarification"
    )


class LeaseAnalysis(BaseModel):
    question: str
    findings: List[ClauseFinding]
    missing_information: List[str]
    summary: str

    @model_validator(mode="before")
    @classmethod
    def normalize_findings(cls, data):
        if not isinstance(data, dict):
            return data

        findings = data.get("findings")

        # Handle Groq returning {"items": [...]}
        # instead of a plain list.
        if isinstance(findings, dict):
            if isinstance(findings.get("items"), list):
                data["findings"] = findings["items"]

        return data
