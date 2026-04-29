"""
Chapter 6: The Council Pattern
===============================
Implementation of multi-agent deliberation for high-stakes decisions.

The council pattern enables multiple agents to:
1. Propose solutions independently
2. Critique each other's proposals
3. Deliberate through structured debate
4. Reach consensus or escalate to human

Use this pattern when single-agent decisions aren't trustworthy enough.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import uuid
from anthropic import AsyncAnthropic


# =============================================================================
# Council Configuration
# =============================================================================

class VotingMethod(str, Enum):
    """Methods for reaching council decisions"""
    UNANIMOUS = "unanimous"      # All must agree
    MAJORITY = "majority"        # >50% must agree
    SUPERMAJORITY = "supermajority"  # >66% must agree
    CONSENSUS = "consensus"      # Iterative refinement until agreement


class ExpertiseArea(str, Enum):
    """Areas of expertise for council members"""
    TECHNICAL = "technical"
    SECURITY = "security"
    BUSINESS = "business"
    LEGAL = "legal"
    ETHICS = "ethics"
    OPERATIONS = "operations"


@dataclass
class CouncilConfig:
    """Configuration for a council"""
    name: str
    description: str
    voting_method: VotingMethod = VotingMethod.MAJORITY
    max_deliberation_rounds: int = 3
    min_confidence_threshold: float = 0.7
    require_human_for_deadlock: bool = True
    require_human_above_impact: str | None = "high"  # low, medium, high, critical


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Proposal:
    """A proposed solution from a council member"""
    id: str
    member_id: str
    content: str
    reasoning: str
    confidence: float
    timestamp: str
    supporting_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass
class Critique:
    """A critique of another member's proposal"""
    id: str
    member_id: str
    target_proposal_id: str
    assessment: Literal["support", "oppose", "neutral"]
    concerns: list[str]
    suggestions: list[str]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "member_id": self.member_id,
            "target_proposal_id": self.target_proposal_id,
            "assessment": self.assessment,
            "concerns": self.concerns,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


@dataclass
class Vote:
    """A vote on a proposal"""
    member_id: str
    proposal_id: str
    vote: Literal["approve", "reject", "abstain"]
    reasoning: str
    timestamp: str


@dataclass
class DeliberationRound:
    """Record of a single deliberation round"""
    round_number: int
    proposals: list[Proposal]
    critiques: list[Critique]
    votes: list[Vote] | None = None
    outcome: str | None = None  # consensus, deadlock, escalate


@dataclass
class CouncilDecision:
    """Final decision from the council"""
    decision_id: str
    question: str
    decision: str
    confidence: float
    voting_result: dict
    deliberation_rounds: list[DeliberationRound]
    dissenting_opinions: list[dict]
    requires_human_approval: bool
    timestamp: str


# =============================================================================
# Council Member Agent
# =============================================================================

class CouncilMember:
    """
    An agent participating in council deliberation.

    Each member has:
    - Specific expertise area
    - Unique perspective/persona
    - Ability to propose, critique, and vote
    """

    def __init__(
        self,
        member_id: str,
        name: str,
        expertise: ExpertiseArea,
        persona: str,
        model: str = "claude-sonnet-4-20250514"
    ):
        self.member_id = member_id
        self.name = name
        self.expertise = expertise
        self.persona = persona
        self.model = model
        self.client = AsyncAnthropic()

    async def close(self):
        """Close the client connection to prevent resource leaks."""
        await self.client.close()

    async def propose(
        self,
        question: str,
        context: dict,
        previous_proposals: list[Proposal] | None = None
    ) -> Proposal:
        """
        Generate a proposal for the question.

        If previous proposals exist (in later rounds), consider them
        when formulating the response.
        """
        system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Your persona: {self.persona}

You are participating in a council deliberation to make an important decision.
Provide your proposal with clear reasoning and confidence level.

Respond in JSON format:
{{
    "proposal": "Your proposed decision/solution",
    "reasoning": "Step-by-step reasoning for your proposal",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", "evidence2"]
}}"""

        user_content = f"Question for deliberation: {question}\n\nContext: {json.dumps(context)}"

        if previous_proposals:
            user_content += "\n\nPrevious proposals from other members:\n"
            for p in previous_proposals:
                user_content += f"- {p.member_id}: {p.content}\n"

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )

        # Parse response
        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {
                "proposal": response.content[0].text,
                "reasoning": "Unable to parse structured response",
                "confidence": 0.5,
                "supporting_evidence": []
            }

        return Proposal(
            id=f"prop_{uuid.uuid4().hex[:8]}",
            member_id=self.member_id,
            content=result.get("proposal", ""),
            reasoning=result.get("reasoning", ""),
            confidence=result.get("confidence", 0.5),
            timestamp=datetime.now(timezone.utc).isoformat(),
            supporting_evidence=result.get("supporting_evidence", [])
        )

    async def critique(
        self,
        proposal: Proposal,
        question: str,
        context: dict
    ) -> Critique:
        """
        Critique another member's proposal.

        Provide assessment, concerns, and constructive suggestions.
        """
        system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Your persona: {self.persona}

You are reviewing another council member's proposal. Provide a fair but thorough critique
from your area of expertise.

Respond in JSON format:
{{
    "assessment": "support" | "oppose" | "neutral",
    "concerns": ["concern1", "concern2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}"""

        user_content = f"""Question: {question}

Proposal from {proposal.member_id}:
{proposal.content}

Their reasoning:
{proposal.reasoning}

Confidence: {proposal.confidence}

Context: {json.dumps(context)}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {
                "assessment": "neutral",
                "concerns": ["Unable to parse structured critique"],
                "suggestions": []
            }

        return Critique(
            id=f"crit_{uuid.uuid4().hex[:8]}",
            member_id=self.member_id,
            target_proposal_id=proposal.id,
            assessment=result.get("assessment", "neutral"),
            concerns=result.get("concerns", []),
            suggestions=result.get("suggestions", []),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    async def vote(
        self,
        proposal: Proposal,
        critiques: list[Critique],
        question: str
    ) -> Vote:
        """
        Vote on a proposal after reviewing all critiques.
        """
        system_prompt = f"""You are {self.name}, a council member with expertise in {self.expertise.value}.

Based on the proposal and all critiques, cast your vote.

Respond in JSON format:
{{
    "vote": "approve" | "reject" | "abstain",
    "reasoning": "Brief explanation for your vote"
}}"""

        critiques_text = "\n".join([
            f"- {c.member_id} ({c.assessment}): {c.concerns}"
            for c in critiques
        ])

        user_content = f"""Question: {question}

Proposal:
{proposal.content}

Critiques:
{critiques_text}"""

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            result = {"vote": "abstain", "reasoning": "Unable to parse vote"}

        return Vote(
            member_id=self.member_id,
            proposal_id=proposal.id,
            vote=result.get("vote", "abstain"),
            reasoning=result.get("reasoning", ""),
            timestamp=datetime.now(timezone.utc).isoformat()
        )


# =============================================================================
# Council Implementation
# =============================================================================

class Council:
    """
    Multi-agent deliberation council for high-stakes decisions.

    The council follows this process:
    1. Each member proposes independently
    2. Members critique each other's proposals
    3. Voting occurs based on configured method
    4. If no consensus, iterate or escalate
    """

    def __init__(
        self,
        config: CouncilConfig,
        members: list[CouncilMember]
    ):
        self.config = config
        self.members = members
        self.deliberation_log: list[DeliberationRound] = []

    async def close(self):
        """Close all member clients to prevent resource leaks."""
        for member in self.members:
            await member.close()

    async def __aenter__(self):
        """Support async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on context exit."""
        await self.close()

    async def deliberate(
        self,
        question: str,
        context: dict,
        impact_level: str = "medium"
    ) -> CouncilDecision:
        """
        Run full deliberation process on a question.

        Args:
            question: The decision question
            context: Relevant context and data
            impact_level: low, medium, high, critical

        Returns:
            CouncilDecision with final outcome and audit trail
        """
        decision_id = f"decision_{uuid.uuid4().hex[:8]}"
        self.deliberation_log = []

        for round_num in range(self.config.max_deliberation_rounds):
            round_result = await self._run_deliberation_round(
                round_num + 1,
                question,
                context
            )
            self.deliberation_log.append(round_result)

            # Check for consensus
            if round_result.outcome == "consensus":
                break
            elif round_result.outcome == "deadlock":
                if self.config.require_human_for_deadlock:
                    return self._create_escalation_decision(
                        decision_id, question, "Deadlock after deliberation"
                    )

        # Final voting
        winning_proposal, voting_result = await self._final_vote(question)

        # Check if human approval required
        requires_human = self._check_human_approval_required(
            impact_level,
            voting_result
        )

        # Collect dissenting opinions
        dissenting = self._collect_dissenting_opinions(
            winning_proposal,
            self.deliberation_log[-1].critiques
        )

        return CouncilDecision(
            decision_id=decision_id,
            question=question,
            decision=winning_proposal.content if winning_proposal else "No consensus reached",
            confidence=winning_proposal.confidence if winning_proposal else 0.0,
            voting_result=voting_result,
            deliberation_rounds=self.deliberation_log,
            dissenting_opinions=dissenting,
            requires_human_approval=requires_human,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    async def _run_deliberation_round(
        self,
        round_number: int,
        question: str,
        context: dict
    ) -> DeliberationRound:
        """Run a single round of deliberation"""

        # Get previous proposals if not first round
        previous_proposals = None
        if round_number > 1 and self.deliberation_log:
            previous_proposals = self.deliberation_log[-1].proposals

        # Phase 1: Gather proposals (parallel)
        proposal_tasks = [
            member.propose(question, context, previous_proposals)
            for member in self.members
        ]
        proposals = await asyncio.gather(*proposal_tasks)

        # Phase 2: Cross-critique (parallel)
        critique_tasks = []
        for member in self.members:
            other_proposals = [p for p in proposals if p.member_id != member.member_id]
            for proposal in other_proposals:
                critique_tasks.append(
                    member.critique(proposal, question, context)
                )

        critiques = await asyncio.gather(*critique_tasks)

        # Determine round outcome
        outcome = self._assess_round_outcome(proposals, critiques)

        return DeliberationRound(
            round_number=round_number,
            proposals=list(proposals),
            critiques=list(critiques),
            outcome=outcome
        )

    def _assess_round_outcome(
        self,
        proposals: list[Proposal],
        critiques: list[Critique]
    ) -> str:
        """Assess whether consensus is emerging"""

        # Count supportive vs opposing critiques
        support_count = sum(1 for c in critiques if c.assessment == "support")
        oppose_count = sum(1 for c in critiques if c.assessment == "oppose")

        total_critiques = len(critiques)
        if total_critiques == 0:
            return "deadlock"

        support_ratio = support_count / total_critiques

        # Check confidence levels
        avg_confidence = sum(p.confidence for p in proposals) / len(proposals)

        if support_ratio > 0.7 and avg_confidence > self.config.min_confidence_threshold:
            return "consensus"
        elif oppose_count > support_count * 2:
            return "deadlock"
        else:
            return "continue"

    async def _final_vote(
        self,
        question: str
    ) -> tuple[Proposal | None, dict]:
        """Conduct final voting on best proposal"""

        if not self.deliberation_log:
            return None, {"error": "No deliberation rounds"}

        last_round = self.deliberation_log[-1]
        proposals = last_round.proposals
        critiques = last_round.critiques

        # Each member votes on each proposal
        all_votes: list[Vote] = []

        for proposal in proposals:
            relevant_critiques = [
                c for c in critiques if c.target_proposal_id == proposal.id
            ]

            vote_tasks = [
                member.vote(proposal, relevant_critiques, question)
                for member in self.members
                if member.member_id != proposal.member_id  # Don't vote on own proposal
            ]

            votes = await asyncio.gather(*vote_tasks)
            all_votes.extend(votes)

        # Tally votes
        vote_counts: dict[str, dict] = {}
        for proposal in proposals:
            proposal_votes = [v for v in all_votes if v.proposal_id == proposal.id]
            vote_counts[proposal.id] = {
                "approve": sum(1 for v in proposal_votes if v.vote == "approve"),
                "reject": sum(1 for v in proposal_votes if v.vote == "reject"),
                "abstain": sum(1 for v in proposal_votes if v.vote == "abstain")
            }

        # Determine winner based on voting method
        winner = self._determine_winner(proposals, vote_counts)

        return winner, vote_counts

    def _determine_winner(
        self,
        proposals: list[Proposal],
        vote_counts: dict[str, dict]
    ) -> Proposal | None:
        """Determine winning proposal based on voting method"""

        total_voters = len(self.members) - 1  # Exclude proposer

        if self.config.voting_method == VotingMethod.UNANIMOUS:
            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] == total_voters:
                    return proposal
            return None

        elif self.config.voting_method == VotingMethod.MAJORITY:
            threshold = total_voters / 2
            best_proposal = None
            best_approval = 0

            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] > threshold and counts["approve"] > best_approval:
                    best_proposal = proposal
                    best_approval = counts["approve"]

            return best_proposal

        elif self.config.voting_method == VotingMethod.SUPERMAJORITY:
            threshold = total_voters * 0.66

            for proposal in proposals:
                counts = vote_counts[proposal.id]
                if counts["approve"] >= threshold:
                    return proposal

            return None

        else:  # CONSENSUS - return highest approval
            best_proposal = max(
                proposals,
                key=lambda p: vote_counts[p.id]["approve"]
            )
            return best_proposal

    def _check_human_approval_required(
        self,
        impact_level: str,
        voting_result: dict
    ) -> bool:
        """Check if decision requires human approval"""

        # Always require human for configured impact level
        impact_hierarchy = ["low", "medium", "high", "critical"]
        if self.config.require_human_above_impact:
            threshold_idx = impact_hierarchy.index(self.config.require_human_above_impact)
            current_idx = impact_hierarchy.index(impact_level)
            if current_idx >= threshold_idx:
                return True

        # Require human if close vote
        for proposal_id, counts in voting_result.items():
            if isinstance(counts, dict):
                approve = counts.get("approve", 0)
                reject = counts.get("reject", 0)
                if approve > 0 and reject > 0:
                    ratio = min(approve, reject) / max(approve, reject)
                    if ratio > 0.6:  # Close vote
                        return True

        return False

    def _collect_dissenting_opinions(
        self,
        winning_proposal: Proposal | None,
        critiques: list[Critique]
    ) -> list[dict]:
        """Collect dissenting opinions for the audit trail"""

        if not winning_proposal:
            return []

        dissenting = []
        for critique in critiques:
            if (critique.target_proposal_id == winning_proposal.id and
                critique.assessment == "oppose"):
                dissenting.append({
                    "member_id": critique.member_id,
                    "concerns": critique.concerns,
                    "suggestions": critique.suggestions
                })

        return dissenting

    def _create_escalation_decision(
        self,
        decision_id: str,
        question: str,
        reason: str
    ) -> CouncilDecision:
        """Create a decision that escalates to human"""
        return CouncilDecision(
            decision_id=decision_id,
            question=question,
            decision=f"ESCALATED: {reason}",
            confidence=0.0,
            voting_result={},
            deliberation_rounds=self.deliberation_log,
            dissenting_opinions=[],
            requires_human_approval=True,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


# =============================================================================
# Pre-built Council Configurations
# =============================================================================

def create_technical_council() -> Council:
    """Create a council for technical decisions"""
    config = CouncilConfig(
        name="Technical Review Council",
        description="Reviews technical decisions and architecture changes",
        voting_method=VotingMethod.MAJORITY,
        max_deliberation_rounds=2,
        require_human_above_impact="high"
    )

    members = [
        CouncilMember(
            member_id="architect",
            name="Alex Architect",
            expertise=ExpertiseArea.TECHNICAL,
            persona="Senior software architect focused on scalability and maintainability"
        ),
        CouncilMember(
            member_id="security",
            name="Sam Security",
            expertise=ExpertiseArea.SECURITY,
            persona="Security engineer focused on threat modeling and secure design"
        ),
        CouncilMember(
            member_id="ops",
            name="Olivia Ops",
            expertise=ExpertiseArea.OPERATIONS,
            persona="SRE focused on reliability, observability, and operational excellence"
        )
    ]

    return Council(config, members)


def create_business_council() -> Council:
    """Create a council for business decisions"""
    config = CouncilConfig(
        name="Business Decision Council",
        description="Reviews business-critical decisions",
        voting_method=VotingMethod.SUPERMAJORITY,
        max_deliberation_rounds=3,
        require_human_above_impact="medium"
    )

    members = [
        CouncilMember(
            member_id="business",
            name="Blake Business",
            expertise=ExpertiseArea.BUSINESS,
            persona="Business strategist focused on ROI and market impact"
        ),
        CouncilMember(
            member_id="legal",
            name="Leslie Legal",
            expertise=ExpertiseArea.LEGAL,
            persona="Legal counsel focused on compliance and risk mitigation"
        ),
        CouncilMember(
            member_id="ethics",
            name="Evan Ethics",
            expertise=ExpertiseArea.ETHICS,
            persona="Ethics advisor focused on fairness and responsible AI"
        )
    ]

    return Council(config, members)


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Demonstrate council deliberation"""

    # Create technical council with context manager to ensure cleanup
    council = create_technical_council()

    # Question for deliberation
    question = """
    Should we migrate our monolithic application to microservices architecture?

    Current state:
    - 500K lines of code
    - 50 developers
    - 99.9% uptime requirement
    - Major feature releases every 2 weeks
    """

    context = {
        "current_architecture": "monolith",
        "team_size": 50,
        "annual_budget": "$5M",
        "risk_tolerance": "medium"
    }

    try:
        print("=" * 60)
        print("Council Deliberation Demo")
        print("=" * 60)
        print(f"\nQuestion: {question[:100]}...")
        print(f"\nCouncil: {council.config.name}")
        print(f"Members: {[m.name for m in council.members]}")
        print(f"Voting Method: {council.config.voting_method.value}")

        # Run deliberation
        print("\nDeliberating...")
        decision = await council.deliberate(
            question=question,
            context=context,
            impact_level="high"
        )

        print("\n" + "=" * 60)
        print("DECISION")
        print("=" * 60)
        print(f"\nDecision ID: {decision.decision_id}")
        print(f"Outcome: {decision.decision}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Requires Human Approval: {decision.requires_human_approval}")

        print(f"\nDeliberation Rounds: {len(decision.deliberation_rounds)}")
        for round_info in decision.deliberation_rounds:
            print(f"  Round {round_info.round_number}: {round_info.outcome}")
            print(f"    Proposals: {len(round_info.proposals)}")
            print(f"    Critiques: {len(round_info.critiques)}")

        if decision.dissenting_opinions:
            print(f"\nDissenting Opinions: {len(decision.dissenting_opinions)}")
            for dissent in decision.dissenting_opinions:
                print(f"  - {dissent['member_id']}: {dissent['concerns'][:2]}")
    finally:
        await council.close()


if __name__ == "__main__":
    asyncio.run(main())
