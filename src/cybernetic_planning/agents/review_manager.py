"""
Economic Plan Review Manager

Central coordinator for the multi - agent economic plan review system.
Manages agent workflows, facilitates communication, and synthesizes final reports.
"""

from typing import Dict, Any, List, Optional
import concurrent.futures
from dataclasses import dataclass, asdict
import time
import json
import logging

from .base import BaseAgent
    EconomicReviewAgent,
    AgentReport,
    CentralPlanningAnalyst,
    LaborValueTheorist,
    MaterialConditionsExpert,
    SocialistDistributionSpecialist,
    ImplementationReviewer,
    WorkersDemocracyExpert,
    SocialDevelopmentAnalyst,
)

# Configure logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReviewSession:
    """Represents a complete economic plan review session."""

    session_id: str
    economic_plan: str
    agent_reports: Dict[str, AgentReport]
    shared_context: Dict[str, Any]
    final_report: Optional[str]
    start_time: float
    end_time: Optional[float]
    status: str  # 'in_progress', 'completed', 'failed'

@dataclass
class ComprehensiveReview:
    """Final comprehensive review of an economic plan."""

    session_id: str
    integrated_summary: str
    cross_domain_analysis: str
    overall_assessment: str
    prioritized_recommendations: List[Dict[str, Any]]
    implementation_roadmap: List[Dict[str, Any]]
    agent_consensus: Dict[str, float]
    confidence_score: float
    timestamp: float

class EconomicPlanReviewManager(BaseAgent):
    """
    Central manager for coordinating economic plan reviews across specialized agents.

    Manages the complete review workflow, facilitates inter - agent communication,
    and synthesizes comprehensive final reports.
    """

    def __init__(self, api_key: str):
        """
        Initialize the review manager.

        Args:
            api_key: Google Gemini API key for all agents
        """
        super().__init__("review_manager", "Economic Plan Review Manager")
        self.api_key = api_key
        self.agents: Dict[str, EconomicReviewAgent] = {}
        self.active_sessions: Dict[str, ReviewSession] = {}
        self.completed_reviews: List[ComprehensiveReview] = []

        # Initialize specialized agents
        self._initialize_agents()

        # Configure Gemini for synthesis tasks

        genai.configure(api_key = api_key)
        self.synthesis_model = genai.GenerativeModel("gemini - 2.0 - flash - exp")

    def _initialize_agents(self):
        """Initialize all specialized review agents."""
        try:
            self.agents = {
                "central_planning": CentralPlanningAnalyst(self.api_key),
                "labor_value": LaborValueTheorist(self.api_key),
                "material_conditions": MaterialConditionsExpert(self.api_key),
                "distribution": SocialistDistributionSpecialist(self.api_key),
                "implementation": ImplementationReviewer(self.api_key),
                "democracy": WorkersDemocracyExpert(self.api_key),
                "social_development": SocialDevelopmentAnalyst(self.api_key),
            }
            logger.info(f"Initialized {len(self.agents)} specialized review agents")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    def start_review_session(self, economic_plan: str, session_id: str = None) -> str:
        """
        Start a new economic plan review session.

        Args:
            economic_plan: The economic plan text to review
            session_id: Optional session ID (auto - generated if not provided)

        Returns:
            Session ID for tracking the review
        """
        if session_id is None:
            session_id = f"review_{int(time.time())}"

        session = ReviewSession(
            session_id = session_id,
            economic_plan = economic_plan,
            agent_reports={},
            shared_context={},
            final_report = None,
            start_time = time.time(),
            end_time = None,
            status="in_progress",
        )

        self.active_sessions[session_id] = session
        logger.info(f"Started review session: {session_id}")

        return session_id

    def conduct_review(self, session_id: str, selected_agents: List[str] = None) -> ComprehensiveReview:
        """
        Conduct a comprehensive review using multiple agents.

        Args:
            session_id: ID of the review session
            selected_agents: List of agent IDs to use (all agents if None)

        Returns:
            Comprehensive review results
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        if selected_agents is None:
            selected_agents = list(self.agents.keys())

        try:
            # Phase 1: Initial individual analysis
            logger.info("Phase 1: Individual agent analysis")
            self._conduct_individual_analysis(session, selected_agents)

            # Phase 2: Cross - agent communication and refinement
            logger.info("Phase 2: Cross - agent communication")
            self._facilitate_cross_agent_communication(session, selected_agents)

            # Phase 3: Synthesis and final report generation
            logger.info("Phase 3: Synthesis and final report")
            comprehensive_review = self._synthesize_final_report(session)

            # Update session status
            session.status = "completed"
            session.end_time = time.time()

            # Store completed review
            self.completed_reviews.append(comprehensive_review)

            logger.info(f"Completed review session: {session_id}")
            return comprehensive_review

        except Exception as e:
            session.status = "failed"
            logger.error(f"Review session {session_id} failed: {str(e)}")
            raise

    def _conduct_individual_analysis(self, session: ReviewSession, selected_agents: List[str]):
        """Conduct individual analysis by each selected agent."""
        with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
            # Submit analysis tasks to all agents
            future_to_agent = {}

            for agent_id in selected_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    future = executor.submit(agent.analyze_plan, session.economic_plan, session.shared_context)
                    future_to_agent[future] = agent_id

            # Collect results
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    report = future.result()
                    session.agent_reports[agent_id] = report
                    logger.info(f"Completed analysis by {agent_id}")
                except Exception as e:
                    logger.error(f"Agent {agent_id} analysis failed: {str(e)}")

    def _facilitate_cross_agent_communication(self, session: ReviewSession, selected_agents: List[str]):
        """Facilitate communication between agents and refine analyses."""
        # Build shared context from all agent reports
        shared_context = {
            "agent_summaries": {},
            "common_themes": [],
            "conflicting_views": [],
            "key_risks": [],
            "recommendations": [],
        }

        # Extract key information from each agent report
        for agent_id, report in session.agent_reports.items():
            shared_context["agent_summaries"][agent_id] = {
                "executive_summary": report.executive_summary,
                "key_recommendations": report.recommendations[:3],
                "confidence": report.confidence_level,
                "main_risks": (
                    report.risk_assessment[:200] + "..."
                    if len(report.risk_assessment) > 200
                    else report.risk_assessment
                ),
            }

            # Collect recommendations
            shared_context["recommendations"].extend(
                [{"agent": agent_id, "recommendation": rec} for rec in report.recommendations]
            )

        # Identify common themes and conflicts using AI
        shared_context.update(self._analyze_agent_consensus(session.agent_reports))

        # Update session shared context
        session.shared_context = shared_context

        # Second round of analysis with shared context (for critical agents)
        critical_agents = ["central_planning", "implementation", "distribution"]
        critical_selected = [a for a in critical_agents if a in selected_agents]

        if critical_selected:
            logger.info("Conducting second round analysis for critical agents")
            with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
                future_to_agent = {}

                for agent_id in critical_selected:
                    agent = self.agents[agent_id]
                    future = executor.submit(agent.analyze_plan, session.economic_plan, shared_context)
                    future_to_agent[future] = agent_id

                # Update reports with refined analysis
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent_id = future_to_agent[future]
                    try:
                        refined_report = future.result()
                        # Merge with original report
                        original_report = session.agent_reports[agent_id]
                        original_report.detailed_analysis += (
                            "\n\nREFINED ANALYSIS:\n" + refined_report.detailed_analysis
                        )
                        original_report.recommendations.extend(refined_report.recommendations)
                        logger.info(f"Refined analysis by {agent_id}")
                    except Exception as e:
                        logger.error(f"Agent {agent_id} refinement failed: {str(e)}")

    def _analyze_agent_consensus(self, agent_reports: Dict[str, AgentReport]) -> Dict[str, Any]:
        """Analyze consensus and conflicts between agent reports using AI."""
        try:
            # Prepare summary of all agent findings
            agent_summaries = []
            for agent_id, report in agent_reports.items():
                agent_summaries.append(
                    f"""
{report.agent_name}:
- Summary: {report.executive_summary}
- Key Risks: {report.risk_assessment[:150]}...
- Top Recommendations: {'; '.join(report.recommendations[:2])}
- Confidence: {report.confidence_level:.2f}
"""
                )

            prompt = f"""
Analyze the following economic plan review findings from multiple specialized agents:

{chr(10).join(agent_summaries)}

Identify:
1. COMMON_THEMES: Key themes that multiple agents agree on
2. CONFLICTING_VIEWS: Areas where agents disagree or have different perspectives
3. KEY_RISKS: Most critical risks identified across agents
4. PRIORITY_AREAS: Areas that need immediate attention based on agent consensus

Format your response as JSON with these four keys.
"""

            response = self.synthesis_model.generate_content(prompt)

            # Parse JSON response
            import re

            json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if json_match:
                consensus_data = json.loads(json_match.group())
                return consensus_data
            else:
                logger.warning("Failed to parse consensus analysis JSON")
                return {}

        except Exception as e:
            logger.error(f"Failed to analyze agent consensus: {str(e)}")
            return {}

    def _synthesize_final_report(self, session: ReviewSession) -> ComprehensiveReview:
        """Synthesize all agent reports into a comprehensive final review."""
        try:
            # Prepare synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(session)

            # Generate comprehensive synthesis
            response = self.synthesis_model.generate_content(synthesis_prompt)

            # Parse synthesis response
            synthesis_sections = self._parse_synthesis_response(response.text)

            # Calculate overall confidence and consensus
            confidence_score = self._calculate_overall_confidence(session.agent_reports)
            agent_consensus = self._calculate_agent_consensus(session.agent_reports)

            # Generate prioritized recommendations
            prioritized_recommendations = self._prioritize_recommendations(session.agent_reports, synthesis_sections)

            # Generate implementation roadmap
            implementation_roadmap = self._generate_implementation_roadmap(synthesis_sections, session.agent_reports)

            comprehensive_review = ComprehensiveReview(
                session_id = session.session_id,
                integrated_summary = synthesis_sections.get("INTEGRATED_SUMMARY", ""),
                cross_domain_analysis = synthesis_sections.get("CROSS_DOMAIN_ANALYSIS", ""),
                overall_assessment = synthesis_sections.get("OVERALL_ASSESSMENT", ""),
                prioritized_recommendations = prioritized_recommendations,
                implementation_roadmap = implementation_roadmap,
                agent_consensus = agent_consensus,
                confidence_score = confidence_score,
                timestamp = time.time(),
            )

            return comprehensive_review

        except Exception as e:
            logger.error(f"Failed to synthesize final report: {str(e)}")
            raise

    def _build_synthesis_prompt(self, session: ReviewSession) -> str:
        """Build prompt for final synthesis."""
        agent_findings = []

        for agent_id, report in session.agent_reports.items():
            agent_findings.append(
                f"""
{report.agent_name} ({report.confidence_level:.2f} confidence):
Executive Summary: {report.executive_summary}
Key Risks: {report.risk_assessment[:300]}...
Top Recommendations: {'; '.join(report.recommendations[:3])}
"""
            )

        return f"""
You are synthesizing a comprehensive economic plan review from multiple specialized agents.

ORIGINAL ECONOMIC PLAN:
{session.economic_plan[:1000]}...

AGENT FINDINGS:
{chr(10).join(agent_findings)}

SHARED CONTEXT:
{json.dumps(session.shared_context, indent = 2)}

Provide a comprehensive synthesis with the following sections:

INTEGRATED_SUMMARY:
[Synthesize key findings across all agents into a coherent 3 - 4 paragraph summary]

CROSS_DOMAIN_ANALYSIS:
[Analyze interactions and dependencies between different economic domains covered by agents]

OVERALL_ASSESSMENT:
[Provide unified evaluation of the economic plan's strengths, weaknesses, and viability]

Focus on:
- Integration of findings across all specialized domains - Identification of cross - cutting issues and synergies - Balanced assessment considering all agent perspectives - Clear, actionable insights for plan improvement
"""

    def _parse_synthesis_response(self, response_text: str) -> Dict[str, str]:
        """Parse synthesis response into sections."""
        sections = {}
        current_section = None
        current_content = []

        lines = response_text.split("\n")

        for line in lines:
            line = line.strip()
            if line.endswith(":") and line.replace(":", "").replace("_", "").isalpha():
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.replace(":", "").upper()
                current_content = []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _calculate_overall_confidence(self, agent_reports: Dict[str, AgentReport]) -> float:
        """Calculate overall confidence score from all agent reports."""
        if not agent_reports:
            return 0.0

        confidence_scores = [report.confidence_level for report in agent_reports.values()]
        return sum(confidence_scores) / len(confidence_scores)

    def _calculate_agent_consensus(self, agent_reports: Dict[str, AgentReport]) -> Dict[str, float]:
        """Calculate consensus metrics between agents."""
        consensus = {}

        for agent_id, report in agent_reports.items():
            consensus[agent_id] = report.confidence_level

        # Add overall consensus score
        confidence_values = list(consensus.values())
        if confidence_values:
            consensus["overall"] = sum(confidence_values) / len(confidence_values)
            consensus["variance"] = sum((x - consensus["overall"]) ** 2 for x in confidence_values) / len(
                confidence_values
            )

        return consensus

    def _prioritize_recommendations(
        self, agent_reports: Dict[str, AgentReport], synthesis_sections: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations from all agent inputs."""
        all_recommendations = []

        # Collect all recommendations with metadata
        for agent_id, report in agent_reports.items():
            for i, rec in enumerate(report.recommendations):
                all_recommendations.append(
                    {
                        "recommendation": rec,
                        "agent": agent_id,
                        "agent_name": report.agent_name,
                        "agent_confidence": report.confidence_level,
                        "priority_score": report.confidence_level
                        * (len(report.recommendations) - i)
                        / len(report.recommendations),
                    }
                )

        # Sort by priority score
        all_recommendations.sort(key = lambda x: x["priority_score"], reverse = True)

        # Return top 10 recommendations
        return all_recommendations[:10]

    def _generate_implementation_roadmap(
        self, synthesis_sections: Dict[str, str], agent_reports: Dict[str, AgentReport]
    ) -> List[Dict[str, Any]]:
        """Generate implementation roadmap based on synthesis and agent reports."""

        # Extract implementation - related recommendations
        impl_recommendations = []
        for agent_id, report in agent_reports.items():
            for rec in report.recommendations:
                if any(keyword in rec.lower() for keyword in ["implement", "establish", "create", "develop", "build"]):
                    impl_recommendations.append(
                        {"action": rec, "agent": report.agent_name, "confidence": report.confidence_level}
                    )

        # Create phased roadmap
        phases = [
            {"phase": "Immediate (0 - 3 months)", "actions": []},
            {"phase": "Short - term (3 - 12 months)", "actions": []},
            {"phase": "Medium - term (1 - 3 years)", "actions": []},
            {"phase": "Long - term (3 + years)", "actions": []},
        ]

        # Distribute recommendations across phases based on urgency keywords
        for rec in impl_recommendations[:12]:  # Limit to 12 actions
            action = rec["action"]
            if any(keyword in action.lower() for keyword in ["immediate", "urgent", "critical"]):
                phases[0]["actions"].append(rec)
            elif any(keyword in action.lower() for keyword in ["short", "quick", "establish"]):
                phases[1]["actions"].append(rec)
            elif any(keyword in action.lower() for keyword in ["develop", "build", "expand"]):
                phases[2]["actions"].append(rec)
            else:
                phases[3]["actions"].append(rec)

        return phases

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a review session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "agents_completed": len(session.agent_reports),
                "total_agents": len(self.agents),
                "has_final_report": session.final_report is not None,
            }
        else:
            return {"error": f"Session {session_id} not found"}

    def export_review(self, session_id: str, format_type: str = "json") -> str:
        """Export a completed review in specified format."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        # Find corresponding comprehensive review
        comprehensive_review = None
        for review in self.completed_reviews:
            if review.session_id == session_id:
                comprehensive_review = review
                break

        if comprehensive_review is None:
            raise ValueError(f"No completed review found for session {session_id}")

        if format_type == "json":
            return self._export_as_json(session, comprehensive_review)
        elif format_type == "text":
            return self._export_as_text(session, comprehensive_review)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _export_as_json(self, session: ReviewSession, review: ComprehensiveReview) -> str:
        """Export review as JSON."""
        export_data = {
            "session": {
                "session_id": session.session_id,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "economic_plan": session.economic_plan,
            },
            "agent_reports": {agent_id: asdict(report) for agent_id, report in session.agent_reports.items()},
            "comprehensive_review": asdict(review),
        }

        return json.dumps(export_data, indent = 2, default = str)

    def _export_as_text(self, session: ReviewSession, review: ComprehensiveReview) -> str:
        """Export review as formatted text."""
        text_report = f"""
ECONOMIC PLAN REVIEW REPORT
==========================

Session ID: {session.session_id}
Review Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(review.timestamp))}
Overall Confidence: {review.confidence_score:.2f}

INTEGRATED SUMMARY
==================
{review.integrated_summary}

CROSS - DOMAIN ANALYSIS
=====================
{review.cross_domain_analysis}

OVERALL ASSESSMENT
==================
{review.overall_assessment}

PRIORITIZED RECOMMENDATIONS
===========================
"""

        for i, rec in enumerate(review.prioritized_recommendations, 1):
            text_report += (
                f"{i}. {rec['recommendation']} (by {rec['agent_name']}, confidence: {rec['agent_confidence']:.2f})\n"
            )

        text_report += "\nIMPLEMENTATION ROADMAP\n=====================\n"

        for phase in review.implementation_roadmap:
            text_report += f"\n{phase['phase']}:\n"
            for action in phase["actions"]:
                text_report += f"  - {action['action']} (by {action['agent']})\n"

        text_report += "\nINDIVIDUAL AGENT REPORTS\n========================\n"

        for agent_id, report in session.agent_reports.items():
            text_report += f"\n{report.agent_name} Report:\n"
            text_report += f"Confidence: {report.confidence_level:.2f}\n"
            text_report += f"Executive Summary: {report.executive_summary}\n"
            text_report += f"Risk Assessment: {report.risk_assessment}\n"
            text_report += "Recommendations:\n"
            for rec in report.recommendations:
                text_report += f"  - {rec}\n"
            text_report += "\n"

        return text_report

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task assigned to the review manager."""
        task_type = task.get("type", "review_plan")

        if task_type == "review_plan":
            economic_plan = task.get("economic_plan", "")
            selected_agents = task.get("selected_agents", None)

            session_id = self.start_review_session(economic_plan)
            comprehensive_review = self.conduct_review(session_id, selected_agents)

            return {
                "status": "completed",
                "session_id": session_id,
                "comprehensive_review": asdict(comprehensive_review),
            }
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def get_capabilities(self) -> List[str]:
        """Get review manager capabilities."""
        return [
            "multi_agent_coordination",
            "economic_plan_review",
            "agent_synthesis",
            "report_generation",
            "workflow_management",
            "cross_agent_communication",
        ]
