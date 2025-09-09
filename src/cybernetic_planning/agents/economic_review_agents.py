"""
Economic Plan Review Agents

Specialized AI agents for comprehensive economic plan analysis using Google Gemini 2.5 Pro.
Each agent focuses on specific aspects of economic planning and socialist theory.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from dataclasses import dataclass
import time
import json
import logging
from .base import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentReport:
    """Structured report from an economic review agent."""
    agent_id: str
    agent_name: str
    executive_summary: str
    detailed_analysis: str
    risk_assessment: str
    recommendations: List[str]
    confidence_level: float  # 0.0 to 1.0
    timestamp: float
    supporting_evidence: List[str]


class EconomicReviewAgent(BaseAgent):
    """
    Base class for all economic plan review agents.
    
    Provides common functionality for AI-powered analysis using Google Gemini 2.5 Pro.
    """
    
    def __init__(self, agent_id: str, name: str, api_key: str, specialization: str):
        """
        Initialize the economic review agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            api_key: Google Gemini API key
            specialization: Agent's area of expertise
        """
        super().__init__(agent_id, name)
        self.api_key = api_key
        self.specialization = specialization
        self.model_name = "gemini-2.0-flash-exp"  # Using latest Gemini model
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Agent-specific system prompt
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        return f"""
You are a specialized economic analyst with expertise in {self.specialization}.
You are part of a multi-agent system reviewing economic plans from a socialist perspective.

Your role is to provide professional, analytical review focusing on {self.specialization}.
Always maintain a formal, academic tone and provide specific, actionable insights.

Key principles:
- Focus on your specialized domain
- Provide evidence-based analysis
- Consider socialist economic theory and principles
- Identify potential risks and opportunities
- Offer concrete recommendations
- Be thorough but concise
- Reference other agents' work when relevant
"""
    
    def analyze_plan(self, economic_plan: str, shared_context: Dict[str, Any] = None) -> AgentReport:
        """
        Analyze an economic plan from this agent's perspective.
        
        Args:
            economic_plan: The economic plan text to analyze
            shared_context: Shared information from other agents
            
        Returns:
            Structured report with analysis results
        """
        try:
            # Build analysis prompt
            prompt = self._build_analysis_prompt(economic_plan, shared_context)
            
            # Generate analysis using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse response into structured report
            report = self._parse_response(response.text)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {str(e)}")
            return self._create_error_report(str(e))
    
    def _build_analysis_prompt(self, economic_plan: str, shared_context: Dict[str, Any] = None) -> str:
        """Build the analysis prompt for Gemini."""
        prompt = f"""
{self.system_prompt}

ECONOMIC PLAN TO ANALYZE:
{economic_plan}

"""
        
        if shared_context:
            prompt += f"""
SHARED CONTEXT FROM OTHER AGENTS:
{json.dumps(shared_context, indent=2)}

"""
        
        prompt += f"""
Please provide a comprehensive analysis of this economic plan from your {self.specialization} perspective.

Structure your response as follows:

EXECUTIVE_SUMMARY:
[Provide a 2-3 sentence summary of your key findings]

DETAILED_ANALYSIS:
[Provide thorough analysis of the plan from your specialized perspective]

RISK_ASSESSMENT:
[Identify potential risks and concerns specific to your domain]

RECOMMENDATIONS:
[List 3-5 specific, actionable recommendations]

CONFIDENCE_LEVEL:
[Rate your confidence in this analysis from 0.0 to 1.0]

SUPPORTING_EVIDENCE:
[List key evidence or data points that support your analysis]
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> AgentReport:
        """Parse Gemini response into structured report."""
        try:
            # Extract sections from response
            sections = self._extract_sections(response_text)
            
            return AgentReport(
                agent_id=self.agent_id,
                agent_name=self.name,
                executive_summary=sections.get('EXECUTIVE_SUMMARY', ''),
                detailed_analysis=sections.get('DETAILED_ANALYSIS', ''),
                risk_assessment=sections.get('RISK_ASSESSMENT', ''),
                recommendations=self._parse_recommendations(sections.get('RECOMMENDATIONS', '')),
                confidence_level=self._parse_confidence(sections.get('CONFIDENCE_LEVEL', '0.5')),
                timestamp=time.time(),
                supporting_evidence=self._parse_evidence(sections.get('SUPPORTING_EVIDENCE', ''))
            )
            
        except Exception as e:
            logger.error(f"Error parsing response from {self.name}: {str(e)}")
            return self._create_error_report(str(e))
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from structured response."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.endswith(':') and line.replace(':', '').replace('_', '').isalpha():
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace(':', '').upper()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_recommendations(self, text: str) -> List[str]:
        """Parse recommendations from text."""
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendations.append(line[1:].strip())
            elif line and len(line) > 10:  # Assume lines with content are recommendations
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _parse_confidence(self, text: str) -> float:
        """Parse confidence level from text."""
        try:
            # Extract numerical value
            import re
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                value = float(match.group(1))
                if value > 1.0:
                    value = value / 100.0  # Convert percentage to decimal
                return min(max(value, 0.0), 1.0)
            return 0.5
        except:
            return 0.5
    
    def _parse_evidence(self, text: str) -> List[str]:
        """Parse supporting evidence from text."""
        evidence = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                evidence.append(line[1:].strip())
            elif line and len(line) > 10:
                evidence.append(line)
        
        return evidence[:10]  # Limit evidence items
    
    def _create_error_report(self, error_message: str) -> AgentReport:
        """Create error report when analysis fails."""
        return AgentReport(
            agent_id=self.agent_id,
            agent_name=self.name,
            executive_summary=f"Analysis failed due to error: {error_message}",
            detailed_analysis="Unable to complete analysis due to technical error.",
            risk_assessment="Cannot assess risks due to analysis failure.",
            recommendations=["Retry analysis with corrected input", "Check API connectivity"],
            confidence_level=0.0,
            timestamp=time.time(),
            supporting_evidence=[]
        )
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task assigned to this agent."""
        task_type = task.get('type', 'analyze_plan')
        
        if task_type == 'analyze_plan':
            plan_text = task.get('economic_plan', '')
            shared_context = task.get('shared_context', {})
            
            report = self.analyze_plan(plan_text, shared_context)
            
            return {
                'status': 'completed',
                'report': report,
                'agent_id': self.agent_id
            }
        else:
            return {'error': f'Unknown task type: {task_type}'}
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [
            f"{self.specialization.lower()}_analysis",
            "economic_plan_review",
            "risk_assessment",
            "recommendation_generation",
            "socialist_theory_application"
        ]


class CentralPlanningAnalyst(EconomicReviewAgent):
    """
    Specialized agent for central planning analysis.
    
    Focuses on production planning, resource allocation, and output targets.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="central_planning_analyst",
            name="Central Planning Analyst",
            api_key=api_key,
            specialization="Central Planning and Resource Allocation"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Production planning and capacity analysis
- Resource allocation optimization
- Output targets and feasibility assessment
- Input-output analysis and sectoral coordination
- Material balance planning
- Capacity utilization and bottleneck identification
- Plan coordination across economic sectors

Focus on:
- Evaluating production targets for realism and achievability
- Assessing resource allocation efficiency
- Identifying potential bottlenecks or coordination issues
- Analyzing sectoral interdependencies
- Reviewing capacity constraints and expansion needs
"""


class LaborValueTheorist(EconomicReviewAgent):
    """
    Specialized agent for labor theory of value analysis.
    
    Focuses on labor value calculations, surplus value, and productivity assessment.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="labor_value_theorist",
            name="Labor Value Theorist",
            api_key=api_key,
            specialization="Labor Theory of Value and Productivity Analysis"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Labor theory of value applications
- Surplus value analysis and distribution
- Productivity measurement and improvement
- Labor time accounting and allocation
- Socially necessary labor time calculations
- Labor intensity analysis across sectors
- Worker productivity and skill development

Focus on:
- Evaluating labor value calculations and methodologies
- Assessing surplus value creation and distribution
- Analyzing productivity trends and improvement potential
- Reviewing labor allocation efficiency
- Identifying opportunities for labor time reduction
- Examining skill development and training needs
"""


class MaterialConditionsExpert(EconomicReviewAgent):
    """
    Specialized agent for material conditions analysis.
    
    Focuses on material dialectics, productive forces, and relations of production.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="material_conditions_expert",
            name="Material Conditions Expert",
            api_key=api_key,
            specialization="Material Dialectics and Productive Forces"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Material dialectics and historical materialism
- Productive forces development and technological progress
- Relations of production analysis
- Infrastructure and material base assessment
- Technology adoption and innovation patterns
- Resource availability and sustainability
- Environmental impact of production

Focus on:
- Analyzing the material foundation of the economic plan
- Evaluating productive forces development
- Assessing technology and infrastructure requirements
- Reviewing environmental sustainability
- Examining resource constraints and availability
- Identifying contradictions between productive forces and relations
"""


class SocialistDistributionSpecialist(EconomicReviewAgent):
    """
    Specialized agent for socialist distribution analysis.
    
    Focuses on "from each according to ability, to each according to need" implementation.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="socialist_distribution_specialist",
            name="Socialist Distribution Specialist",
            api_key=api_key,
            specialization="Socialist Distribution and Social Needs"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Socialist distribution principles and mechanisms
- Social needs assessment and prioritization
- Public goods provision and accessibility
- Income distribution and inequality reduction
- Social services planning and delivery
- Universal basic services implementation
- Community needs analysis

Focus on:
- Evaluating distribution mechanisms for fairness and efficiency
- Assessing social needs coverage and prioritization
- Analyzing public goods and services provision
- Reviewing accessibility and universal access
- Identifying gaps in social needs fulfillment
- Examining community participation in distribution decisions
"""


class ImplementationReviewer(EconomicReviewAgent):
    """
    Specialized agent for implementation feasibility analysis.
    
    Focuses on feasibility, timeline, and resource coordination in planned economy.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="implementation_reviewer",
            name="Implementation Reviewer",
            api_key=api_key,
            specialization="Implementation Feasibility and Coordination"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Implementation feasibility assessment
- Timeline analysis and milestone planning
- Resource coordination and logistics
- Institutional capacity evaluation
- Risk management and contingency planning
- Monitoring and evaluation frameworks
- Administrative and organizational requirements

Focus on:
- Evaluating implementation feasibility and realistic timelines
- Assessing resource coordination requirements
- Analyzing institutional capacity and readiness
- Identifying implementation risks and mitigation strategies
- Reviewing monitoring and evaluation mechanisms
- Examining administrative and organizational structures needed
"""


class WorkersDemocracyExpert(EconomicReviewAgent):
    """
    Specialized agent for workers' democracy analysis.
    
    Focuses on democratic participation, worker control, and collective decision-making.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="workers_democracy_expert",
            name="Workers' Democracy Expert",
            api_key=api_key,
            specialization="Workers' Democracy and Collective Decision-Making"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Workers' democracy and participation mechanisms
- Collective decision-making processes
- Worker control and workplace democracy
- Democratic planning and community involvement
- Participatory budgeting and resource allocation
- Worker representation and voice
- Democratic institutions and governance structures

Focus on:
- Evaluating democratic participation mechanisms in the plan
- Assessing worker control and decision-making power
- Analyzing collective decision-making processes
- Reviewing community involvement and representation
- Identifying opportunities for increased democratic participation
- Examining governance structures for democratic accountability
"""


class SocialDevelopmentAnalyst(EconomicReviewAgent):
    """
    Specialized agent for social development analysis.
    
    Focuses on meeting social needs, eliminating exploitation, and class analysis.
    """
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="social_development_analyst",
            name="Social Development Analyst",
            api_key=api_key,
            specialization="Social Development and Class Analysis"
        )
    
    def _build_system_prompt(self) -> str:
        base_prompt = super()._build_system_prompt()
        return base_prompt + """

Your specific expertise includes:
- Social development and human welfare analysis
- Class structure and social stratification
- Exploitation elimination and worker empowerment
- Social mobility and equality promotion
- Education, healthcare, and social services
- Cultural development and social progress
- Community development and social cohesion

Focus on:
- Evaluating social development outcomes and targets
- Analyzing class relations and power structures
- Assessing exploitation elimination measures
- Reviewing social services and welfare provision
- Identifying opportunities for social progress
- Examining community development and empowerment initiatives
"""