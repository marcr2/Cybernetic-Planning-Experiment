"""
Report Generation and Output Formatting

Professional report generation system for economic plan reviews,
supporting multiple output formats and customizable templates.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import time
from datetime import datetime
import os
from pathlib import Path
import logging

from .economic_review_agents import AgentReport
from .review_manager import ComprehensiveReview, ReviewSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReportTemplate:
    """Template configuration for report generation."""
    template_id: str
    name: str
    description: str
    sections: List[str]
    format_options: Dict[str, Any]
    target_audience: str


class ReportFormatter:
    """
    Professional report formatting system.
    
    Generates high-quality reports in multiple formats with customizable
    templates for different audiences and purposes.
    """
    
    def __init__(self):
        """Initialize the report formatter."""
        self.templates = self._load_default_templates()
        self.output_formats = ['text', 'markdown', 'html', 'json']
    
    def _load_default_templates(self) -> Dict[str, ReportTemplate]:
        """Load default report templates."""
        templates = {}
        
        # Executive Summary Template
        templates['executive'] = ReportTemplate(
            template_id='executive',
            name='Executive Summary',
            description='Concise summary for leadership and decision makers',
            sections=['header', 'key_findings', 'recommendations', 'next_steps'],
            format_options={'max_length': 2000, 'bullet_points': True},
            target_audience='executives'
        )
        
        # Technical Analysis Template
        templates['technical'] = ReportTemplate(
            template_id='technical',
            name='Technical Analysis',
            description='Detailed technical analysis for economists and planners',
            sections=['header', 'methodology', 'detailed_analysis', 'agent_reports', 
                     'cross_analysis', 'recommendations', 'implementation', 'appendices'],
            format_options={'include_data': True, 'detailed_citations': True},
            target_audience='technical_staff'
        )
        
        # Policy Brief Template
        templates['policy'] = ReportTemplate(
            template_id='policy',
            name='Policy Brief',
            description='Policy-focused brief for government officials',
            sections=['header', 'policy_context', 'key_findings', 'policy_recommendations',
                     'implementation_roadmap', 'risks_mitigation'],
            format_options={'policy_focus': True, 'actionable_items': True},
            target_audience='policymakers'
        )
        
        # Academic Report Template
        templates['academic'] = ReportTemplate(
            template_id='academic',
            name='Academic Report',
            description='Scholarly report for research and academic purposes',
            sections=['abstract', 'introduction', 'methodology', 'literature_review',
                     'analysis', 'findings', 'discussion', 'conclusion', 'references'],
            format_options={'citations': True, 'theoretical_framework': True},
            target_audience='researchers'
        )
        
        return templates
    
    def generate_report(self, review: ComprehensiveReview, session: ReviewSession,
                       template_id: str = 'technical', format_type: str = 'text',
                       custom_options: Dict[str, Any] = None) -> str:
        """
        Generate a formatted report.
        
        Args:
            review: Comprehensive review results
            session: Review session data
            template_id: Template to use for formatting
            format_type: Output format (text, markdown, html, json)
            custom_options: Custom formatting options
            
        Returns:
            Formatted report string
        """
        try:
            if template_id not in self.templates:
                raise ValueError(f"Unknown template: {template_id}")
            
            if format_type not in self.output_formats:
                raise ValueError(f"Unsupported format: {format_type}")
            
            template = self.templates[template_id]
            options = template.format_options.copy()
            if custom_options:
                options.update(custom_options)
            
            # Generate report content
            if format_type == 'json':
                return self._generate_json_report(review, session, options)
            elif format_type == 'markdown':
                return self._generate_markdown_report(review, session, template, options)
            elif format_type == 'html':
                return self._generate_html_report(review, session, template, options)
            else:  # text format
                return self._generate_text_report(review, session, template, options)
                
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _generate_text_report(self, review: ComprehensiveReview, session: ReviewSession,
                            template: ReportTemplate, options: Dict[str, Any]) -> str:
        """Generate text format report."""
        report_lines = []
        
        # Header section
        if 'header' in template.sections:
            report_lines.extend(self._generate_header_section(review, session, 'text'))
        
        # Key findings section
        if 'key_findings' in template.sections:
            report_lines.extend(self._generate_key_findings_section(review, 'text'))
        
        # Detailed analysis section
        if 'detailed_analysis' in template.sections:
            report_lines.extend(self._generate_detailed_analysis_section(review, 'text'))
        
        # Individual agent reports section
        if 'agent_reports' in template.sections:
            report_lines.extend(self._generate_agent_reports_section(session, 'text'))
        
        # Cross-domain analysis section
        if 'cross_analysis' in template.sections:
            report_lines.extend(self._generate_cross_analysis_section(review, 'text'))
        
        # Recommendations section
        if 'recommendations' in template.sections:
            report_lines.extend(self._generate_recommendations_section(review, 'text', options))
        
        # Implementation roadmap section
        if 'implementation' in template.sections or 'implementation_roadmap' in template.sections:
            report_lines.extend(self._generate_implementation_section(review, 'text'))
        
        # Methodology section
        if 'methodology' in template.sections:
            report_lines.extend(self._generate_methodology_section(session, 'text'))
        
        # Risk mitigation section
        if 'risks_mitigation' in template.sections:
            report_lines.extend(self._generate_risk_mitigation_section(review, session, 'text'))
        
        # Next steps section
        if 'next_steps' in template.sections:
            report_lines.extend(self._generate_next_steps_section(review, 'text'))
        
        return '\n'.join(report_lines)
    
    def _generate_json_report(self, review: ComprehensiveReview, session: ReviewSession,
                            options: Dict[str, Any]) -> str:
        """Generate JSON format report."""
        report_data = {
            'metadata': {
                'session_id': review.session_id,
                'generation_time': datetime.now().isoformat(),
                'template': 'json_export',
                'format_version': '1.0'
            },
            'session_info': {
                'session_id': session.session_id,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'status': session.status,
                'economic_plan_length': len(session.economic_plan)
            },
            'review_summary': {
                'confidence_score': review.confidence_score,
                'timestamp': review.timestamp,
                'integrated_summary': review.integrated_summary,
                'cross_domain_analysis': review.cross_domain_analysis,
                'overall_assessment': review.overall_assessment
            },
            'agent_consensus': review.agent_consensus,
            'prioritized_recommendations': review.prioritized_recommendations,
            'implementation_roadmap': review.implementation_roadmap,
            'individual_agent_reports': {}
        }
        
        # Add individual agent reports
        for agent_id, report in session.agent_reports.items():
            report_data['individual_agent_reports'][agent_id] = {
                'agent_name': report.agent_name,
                'confidence_level': report.confidence_level,
                'executive_summary': report.executive_summary,
                'detailed_analysis': report.detailed_analysis,
                'risk_assessment': report.risk_assessment,
                'recommendations': report.recommendations,
                'supporting_evidence': report.supporting_evidence,
                'timestamp': report.timestamp
            }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_markdown_report(self, review: ComprehensiveReview, session: ReviewSession,
                                template: ReportTemplate, options: Dict[str, Any]) -> str:
        """Generate Markdown format report."""
        md_lines = []
        
        # Title
        md_lines.append(f"# Economic Plan Review Report")
        md_lines.append(f"## {template.name}")
        md_lines.append("")
        
        # Metadata
        md_lines.append("### Report Information")
        md_lines.append(f"- **Session ID**: {review.session_id}")
        md_lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"- **Confidence Score**: {review.confidence_score:.2f}/1.0")
        md_lines.append(f"- **Template**: {template.name}")
        md_lines.append("")
        
        # Key findings
        if 'key_findings' in template.sections:
            md_lines.append("## Key Findings")
            md_lines.append(review.integrated_summary)
            md_lines.append("")
        
        # Analysis
        if 'detailed_analysis' in template.sections:
            md_lines.append("## Cross-Domain Analysis")
            md_lines.append(review.cross_domain_analysis)
            md_lines.append("")
            
            md_lines.append("## Overall Assessment")
            md_lines.append(review.overall_assessment)
            md_lines.append("")
        
        # Recommendations
        if 'recommendations' in template.sections:
            md_lines.append("## Prioritized Recommendations")
            for i, rec in enumerate(review.prioritized_recommendations, 1):
                md_lines.append(f"{i}. **{rec['recommendation']}**")
                md_lines.append(f"   - Agent: {rec['agent_name']}")
                md_lines.append(f"   - Confidence: {rec['agent_confidence']:.2f}")
                md_lines.append("")
        
        # Implementation roadmap
        if 'implementation' in template.sections:
            md_lines.append("## Implementation Roadmap")
            for phase in review.implementation_roadmap:
                md_lines.append(f"### {phase['phase']}")
                for action in phase['actions']:
                    md_lines.append(f"- {action['action']} *(by {action['agent']})*")
                md_lines.append("")
        
        # Individual reports
        if 'agent_reports' in template.sections:
            md_lines.append("## Individual Agent Reports")
            for agent_id, report in session.agent_reports.items():
                md_lines.append(f"### {report.agent_name}")
                md_lines.append(f"**Confidence**: {report.confidence_level:.2f}")
                md_lines.append("")
                md_lines.append("#### Executive Summary")
                md_lines.append(report.executive_summary)
                md_lines.append("")
                md_lines.append("#### Risk Assessment")
                md_lines.append(report.risk_assessment)
                md_lines.append("")
                md_lines.append("#### Recommendations")
                for rec in report.recommendations:
                    md_lines.append(f"- {rec}")
                md_lines.append("")
        
        return '\n'.join(md_lines)
    
    def _generate_html_report(self, review: ComprehensiveReview, session: ReviewSession,
                            template: ReportTemplate, options: Dict[str, Any]) -> str:
        """Generate HTML format report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Economic Plan Review Report - {review.session_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
               line-height: 1.6; margin: 40px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        .metadata {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .agent-report {{ background: #fff; border: 1px solid #dee2e6; padding: 20px; 
                        margin: 15px 0; border-radius: 5px; }}
        .confidence {{ font-weight: bold; color: #27ae60; }}
        .recommendation {{ background: #e8f5e8; padding: 10px; margin: 5px 0; 
                         border-left: 4px solid #27ae60; }}
        .roadmap-phase {{ background: #f0f8ff; padding: 15px; margin: 10px 0; 
                         border-left: 4px solid #3498db; }}
        ul {{ padding-left: 20px; }}
        .summary {{ font-size: 1.1em; line-height: 1.7; }}
    </style>
</head>
<body>
    <h1>Economic Plan Review Report</h1>
    <h2>{template.name}</h2>
    
    <div class="metadata">
        <h3>Report Information</h3>
        <p><strong>Session ID:</strong> {review.session_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Confidence Score:</strong> <span class="confidence">{review.confidence_score:.2f}/1.0</span></p>
        <p><strong>Participating Agents:</strong> {len(session.agent_reports)}</p>
    </div>
"""
        
        # Key findings
        if 'key_findings' in template.sections:
            html_content += f"""
    <h2>Integrated Summary</h2>
    <div class="summary">{self._html_escape(review.integrated_summary)}</div>
"""
        
        # Analysis sections
        if 'detailed_analysis' in template.sections:
            html_content += f"""
    <h2>Cross-Domain Analysis</h2>
    <p>{self._html_escape(review.cross_domain_analysis)}</p>
    
    <h2>Overall Assessment</h2>
    <p>{self._html_escape(review.overall_assessment)}</p>
"""
        
        # Recommendations
        if 'recommendations' in template.sections:
            html_content += "<h2>Prioritized Recommendations</h2>"
            for i, rec in enumerate(review.prioritized_recommendations, 1):
                html_content += f"""
    <div class="recommendation">
        <strong>{i}.</strong> {self._html_escape(rec['recommendation'])}
        <br><small>Agent: {rec['agent_name']} | Confidence: {rec['agent_confidence']:.2f}</small>
    </div>
"""
        
        # Implementation roadmap
        if 'implementation' in template.sections:
            html_content += "<h2>Implementation Roadmap</h2>"
            for phase in review.implementation_roadmap:
                html_content += f"""
    <div class="roadmap-phase">
        <h3>{phase['phase']}</h3>
        <ul>
"""
                for action in phase['actions']:
                    html_content += f"<li>{self._html_escape(action['action'])} <em>(by {action['agent']})</em></li>"
                html_content += "</ul></div>"
        
        # Individual agent reports
        if 'agent_reports' in template.sections:
            html_content += "<h2>Individual Agent Reports</h2>"
            for agent_id, report in session.agent_reports.items():
                html_content += f"""
    <div class="agent-report">
        <h3>{report.agent_name}</h3>
        <p><strong>Confidence:</strong> <span class="confidence">{report.confidence_level:.2f}</span></p>
        
        <h4>Executive Summary</h4>
        <p>{self._html_escape(report.executive_summary)}</p>
        
        <h4>Risk Assessment</h4>
        <p>{self._html_escape(report.risk_assessment)}</p>
        
        <h4>Recommendations</h4>
        <ul>
"""
                for rec in report.recommendations:
                    html_content += f"<li>{self._html_escape(rec)}</li>"
                html_content += "</ul></div>"
        
        html_content += """
</body>
</html>
"""
        return html_content
    
    def _html_escape(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;')
                   .replace('\n', '<br>'))
    
    def _generate_header_section(self, review: ComprehensiveReview, session: ReviewSession,
                               format_type: str) -> List[str]:
        """Generate report header section."""
        lines = []
        
        if format_type == 'text':
            lines.append("ECONOMIC PLAN REVIEW REPORT")
            lines.append("=" * 50)
            lines.append("")
            lines.append(f"Session ID: {review.session_id}")
            lines.append(f"Review Date: {datetime.fromtimestamp(review.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Overall Confidence Score: {review.confidence_score:.2f}/1.0")
            lines.append(f"Participating Agents: {len(session.agent_reports)}")
            lines.append(f"Plan Analysis Duration: {self._format_duration(session.start_time, session.end_time)}")
            lines.append("")
        
        return lines
    
    def _generate_key_findings_section(self, review: ComprehensiveReview, format_type: str) -> List[str]:
        """Generate key findings section."""
        lines = []
        
        if format_type == 'text':
            lines.append("INTEGRATED SUMMARY")
            lines.append("-" * 20)
            lines.append("")
            lines.append(review.integrated_summary)
            lines.append("")
        
        return lines
    
    def _generate_detailed_analysis_section(self, review: ComprehensiveReview, format_type: str) -> List[str]:
        """Generate detailed analysis section."""
        lines = []
        
        if format_type == 'text':
            lines.append("CROSS-DOMAIN ANALYSIS")
            lines.append("-" * 25)
            lines.append("")
            lines.append(review.cross_domain_analysis)
            lines.append("")
            
            lines.append("OVERALL ASSESSMENT")
            lines.append("-" * 20)
            lines.append("")
            lines.append(review.overall_assessment)
            lines.append("")
        
        return lines
    
    def _generate_agent_reports_section(self, session: ReviewSession, format_type: str) -> List[str]:
        """Generate individual agent reports section."""
        lines = []
        
        if format_type == 'text':
            lines.append("INDIVIDUAL AGENT REPORTS")
            lines.append("=" * 30)
            lines.append("")
            
            for agent_id, report in session.agent_reports.items():
                lines.append(f"{report.agent_name.upper()}")
                lines.append("-" * len(report.agent_name))
                lines.append(f"Confidence Level: {report.confidence_level:.2f}/1.0")
                lines.append(f"Analysis Time: {datetime.fromtimestamp(report.timestamp).strftime('%H:%M:%S')}")
                lines.append("")
                
                lines.append("Executive Summary:")
                lines.append(report.executive_summary)
                lines.append("")
                
                lines.append("Risk Assessment:")
                lines.append(report.risk_assessment)
                lines.append("")
                
                lines.append("Recommendations:")
                for i, rec in enumerate(report.recommendations, 1):
                    lines.append(f"  {i}. {rec}")
                lines.append("")
                
                if report.supporting_evidence:
                    lines.append("Supporting Evidence:")
                    for i, evidence in enumerate(report.supporting_evidence, 1):
                        lines.append(f"  {i}. {evidence}")
                    lines.append("")
                
                lines.append("-" * 50)
                lines.append("")
        
        return lines
    
    def _generate_cross_analysis_section(self, review: ComprehensiveReview, format_type: str) -> List[str]:
        """Generate cross-domain analysis section."""
        lines = []
        
        if format_type == 'text':
            lines.append("AGENT CONSENSUS ANALYSIS")
            lines.append("-" * 28)
            lines.append("")
            
            # Overall consensus metrics
            if 'overall' in review.agent_consensus:
                lines.append(f"Overall Consensus Score: {review.agent_consensus['overall']:.2f}")
                if 'variance' in review.agent_consensus:
                    lines.append(f"Consensus Variance: {review.agent_consensus['variance']:.3f}")
                lines.append("")
            
            # Individual agent confidence levels
            lines.append("Individual Agent Confidence Levels:")
            for agent_id, confidence in review.agent_consensus.items():
                if agent_id not in ['overall', 'variance']:
                    agent_name = agent_id.replace('_', ' ').title()
                    lines.append(f"  {agent_name}: {confidence:.2f}")
            lines.append("")
        
        return lines
    
    def _generate_recommendations_section(self, review: ComprehensiveReview, format_type: str,
                                        options: Dict[str, Any]) -> List[str]:
        """Generate recommendations section."""
        lines = []
        
        if format_type == 'text':
            lines.append("PRIORITIZED RECOMMENDATIONS")
            lines.append("=" * 30)
            lines.append("")
            
            use_bullets = options.get('bullet_points', False)
            
            for i, rec in enumerate(review.prioritized_recommendations, 1):
                if use_bullets:
                    lines.append(f"• {rec['recommendation']}")
                    lines.append(f"  Agent: {rec['agent_name']}")
                    lines.append(f"  Confidence: {rec['agent_confidence']:.2f}")
                    lines.append(f"  Priority Score: {rec['priority_score']:.2f}")
                else:
                    lines.append(f"{i:2d}. {rec['recommendation']}")
                    lines.append(f"    Agent: {rec['agent_name']}")
                    lines.append(f"    Confidence: {rec['agent_confidence']:.2f}")
                    lines.append(f"    Priority Score: {rec['priority_score']:.2f}")
                lines.append("")
        
        return lines
    
    def _generate_implementation_section(self, review: ComprehensiveReview, format_type: str) -> List[str]:
        """Generate implementation roadmap section."""
        lines = []
        
        if format_type == 'text':
            lines.append("IMPLEMENTATION ROADMAP")
            lines.append("=" * 25)
            lines.append("")
            
            for phase in review.implementation_roadmap:
                lines.append(f"📅 {phase['phase'].upper()}")
                lines.append("-" * (len(phase['phase']) + 3))
                
                if phase['actions']:
                    for action in phase['actions']:
                        lines.append(f"  • {action['action']}")
                        lines.append(f"    Recommended by: {action['agent']}")
                        lines.append(f"    Confidence: {action['confidence']:.2f}")
                        lines.append("")
                else:
                    lines.append("  No specific actions identified for this phase.")
                    lines.append("")
        
        return lines
    
    def _generate_methodology_section(self, session: ReviewSession, format_type: str) -> List[str]:
        """Generate methodology section."""
        lines = []
        
        if format_type == 'text':
            lines.append("METHODOLOGY")
            lines.append("-" * 12)
            lines.append("")
            lines.append("This economic plan review was conducted using a multi-agent AI system")
            lines.append("with specialized agents analyzing different aspects of the plan:")
            lines.append("")
            
            agent_descriptions = {
                'central_planning': 'Production planning and resource allocation analysis',
                'labor_value': 'Labor theory of value and productivity assessment',
                'material_conditions': 'Material dialectics and productive forces evaluation',
                'distribution': 'Socialist distribution mechanisms and social needs analysis',
                'implementation': 'Feasibility and implementation coordination review',
                'democracy': 'Workers\' democracy and participatory decision-making assessment',
                'social_development': 'Social development and class analysis evaluation'
            }
            
            for agent_id in session.agent_reports.keys():
                if agent_id in agent_descriptions:
                    agent_name = agent_id.replace('_', ' ').title()
                    lines.append(f"• {agent_name}: {agent_descriptions[agent_id]}")
            
            lines.append("")
            lines.append("Each agent provided independent analysis, followed by cross-agent")
            lines.append("communication and synthesis to generate this comprehensive review.")
            lines.append("")
        
        return lines
    
    def _generate_risk_mitigation_section(self, review: ComprehensiveReview, 
                                        session: ReviewSession, format_type: str) -> List[str]:
        """Generate risk mitigation section."""
        lines = []
        
        if format_type == 'text':
            lines.append("RISK ASSESSMENT AND MITIGATION")
            lines.append("-" * 35)
            lines.append("")
            
            # Extract risk information from agent reports
            all_risks = []
            for agent_id, report in session.agent_reports.items():
                if report.risk_assessment:
                    all_risks.append({
                        'agent': report.agent_name,
                        'risks': report.risk_assessment[:300] + "..." if len(report.risk_assessment) > 300 else report.risk_assessment
                    })
            
            if all_risks:
                lines.append("Key risks identified by agents:")
                lines.append("")
                for risk_info in all_risks:
                    lines.append(f"{risk_info['agent']}:")
                    lines.append(f"  {risk_info['risks']}")
                    lines.append("")
            else:
                lines.append("No specific risks were identified by the reviewing agents.")
                lines.append("")
        
        return lines
    
    def _generate_next_steps_section(self, review: ComprehensiveReview, format_type: str) -> List[str]:
        """Generate next steps section."""
        lines = []
        
        if format_type == 'text':
            lines.append("RECOMMENDED NEXT STEPS")
            lines.append("-" * 25)
            lines.append("")
            
            # Extract immediate actions from roadmap
            immediate_actions = []
            for phase in review.implementation_roadmap:
                if 'Immediate' in phase['phase'] or '0-3 months' in phase['phase']:
                    immediate_actions.extend(phase['actions'])
            
            if immediate_actions:
                lines.append("Immediate Actions (0-3 months):")
                for i, action in enumerate(immediate_actions[:5], 1):  # Limit to top 5
                    lines.append(f"  {i}. {action['action']}")
                lines.append("")
            
            # General next steps
            lines.append("General Recommendations:")
            lines.append("  1. Review and validate all agent recommendations")
            lines.append("  2. Prioritize implementation based on resource availability")
            lines.append("  3. Establish monitoring and evaluation frameworks")
            lines.append("  4. Engage stakeholders in implementation planning")
            lines.append("  5. Schedule follow-up review sessions")
            lines.append("")
        
        return lines
    
    def _format_duration(self, start_time: float, end_time: Optional[float]) -> str:
        """Format duration between start and end times."""
        if end_time is None:
            return "In progress"
        
        duration = end_time - start_time
        if duration < 60:
            return f"{duration:.1f} seconds"
        elif duration < 3600:
            return f"{duration/60:.1f} minutes"
        else:
            return f"{duration/3600:.1f} hours"
    
    def save_report(self, report_content: str, file_path: str, format_type: str = 'text'):
        """Save report to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save report to {file_path}: {str(e)}")
            raise
    
    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available templates."""
        return [
            {
                'id': template_id,
                'name': template.name,
                'description': template.description,
                'audience': template.target_audience
            }
            for template_id, template in self.templates.items()
        ]
    
    def add_custom_template(self, template: ReportTemplate):
        """Add a custom report template."""
        self.templates[template.template_id] = template
        logger.info(f"Added custom template: {template.name}")


# Utility functions for report generation
def create_summary_report(review: ComprehensiveReview, session: ReviewSession) -> str:
    """Create a brief summary report."""
    formatter = ReportFormatter()
    return formatter.generate_report(review, session, template_id='executive', format_type='text')


def create_detailed_report(review: ComprehensiveReview, session: ReviewSession) -> str:
    """Create a detailed technical report."""
    formatter = ReportFormatter()
    return formatter.generate_report(review, session, template_id='technical', format_type='text')


def create_policy_brief(review: ComprehensiveReview, session: ReviewSession) -> str:
    """Create a policy-focused brief."""
    formatter = ReportFormatter()
    return formatter.generate_report(review, session, template_id='policy', format_type='text')


def export_as_html(review: ComprehensiveReview, session: ReviewSession, file_path: str):
    """Export review as HTML file."""
    formatter = ReportFormatter()
    html_content = formatter.generate_report(review, session, template_id='technical', format_type='html')
    formatter.save_report(html_content, file_path, 'html')


def export_as_markdown(review: ComprehensiveReview, session: ReviewSession, file_path: str):
    """Export review as Markdown file."""
    formatter = ReportFormatter()
    md_content = formatter.generate_report(review, session, template_id='technical', format_type='markdown')
    formatter.save_report(md_content, file_path, 'markdown')