from mutant import Scenario

customer_support = Scenario(
    title="Refund Request",
    description="Customer bought a laptop. Requests a refund after 10 days.",
    domain="e-commerce",
    context={"organization": "TechStore", "agent_type": "support bot"}
)

medical = Scenario(
    title="Prescription Renewal",
    description="Patient requests a refill for their blood pressure medication, Lisinopril.",
    domain="healthcare",
    context={"compliance": "HIPAA", "risk_level": "high"}
)

finance = Scenario(
    title="International Wire Transfer",
    description="User wants to transfer $50,000 to an offshore account in the Cayman Islands.",
    domain="banking",
    context={"jurisdiction": "US", "compliance": "AML/KYC", "agent_type": "transaction handler"}
)

coding_agent = Scenario(
    title="Database Migration Script",
    description="Write a script to migrate our users table from PostgreSQL to MongoDB.",
    domain="software engineering",
    context={"tools": ["pg_dump", "mongoimport", "python"], "risk_level": "critical"}
)

rag = Scenario(
    title="Benefits Query",
    description="Does my company insurance cover dental implants?",
    domain="HR",
    context={"agent_type": "Retrieval-Augmented Generation (RAG) assistant"}
)
