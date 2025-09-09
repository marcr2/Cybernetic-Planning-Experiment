# Economic Plan Review System - Multi-Agent AI Analysis

A comprehensive multi-agent AI system for reviewing and analyzing economic plans through specialized AI agents using Google Gemini 2.5 Pro API. Each agent brings expertise in different aspects of socialist economic theory and planning.

## 🌟 Features

### Multi-Agent Analysis System
- **7 Specialized AI Agents** each with domain expertise
- **Collaborative Analysis** with inter-agent communication
- **Professional Reports** with comprehensive findings
- **Cross-Domain Synthesis** integrating all perspectives

### Specialized Economic Agents

#### 🏭 Central Planning Analyst
- Production planning and capacity analysis
- Resource allocation optimization
- Output targets and feasibility assessment
- Sectoral coordination and material balance

#### ⚒️ Labor Value Theorist
- Labor theory of value applications
- Surplus value analysis and distribution
- Productivity measurement and improvement
- Socially necessary labor time calculations

#### 🔬 Material Conditions Expert
- Material dialectics and historical materialism
- Productive forces development
- Relations of production analysis
- Infrastructure and technological assessment

#### 🤝 Socialist Distribution Specialist
- "From each according to ability, to each according to need"
- Social needs assessment and prioritization
- Public goods provision and accessibility
- Universal basic services implementation

#### ⚙️ Implementation Reviewer
- Implementation feasibility assessment
- Timeline analysis and milestone planning
- Resource coordination and logistics
- Risk management and contingency planning

#### 🗳️ Workers' Democracy Expert
- Democratic participation mechanisms
- Worker control and workplace democracy
- Collective decision-making processes
- Participatory budgeting and resource allocation

#### 🌱 Social Development Analyst
- Social development and human welfare
- Class structure and social stratification
- Exploitation elimination measures
- Education, healthcare, and social services

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Google Gemini 2.5 Pro API key
- Internet connection
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd economic-plan-review-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Get Google Gemini API Key:**
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Sign in with your Google account
   - Create an API key
   - Keep it secure and private

4. **Run the application:**
```bash
python economic_plan_review_gui.py
```

## 📋 Usage Guide

### Step 1: Configure API Key
1. Open the "🔑 API Configuration" tab
2. Enter your Google Gemini 2.5 Pro API key
3. Click "Test API Key" to verify
4. Click "Save API Key" to store securely

### Step 2: Load Economic Plan
1. Go to "📄 Plan Input" tab
2. Load from file (PDF, DOCX, TXT) or paste text
3. Review the plan text and character count

### Step 3: Select Agents
1. Open "🤖 Agent Selection" tab
2. Choose which specialized agents to include
3. Review agent descriptions and capabilities

### Step 4: Start Review
1. Switch to "⚙️ Review Progress" tab
2. Click "🚀 Start Review"
3. Monitor progress and agent status

### Step 5: Review Results
1. Go to "📊 Results" tab
2. Read the integrated summary
3. Review individual agent reports
4. Check prioritized recommendations
5. Examine implementation roadmap

### Step 6: Export Results
1. Open "💾 Export" tab
2. Choose format (Text, JSON, Summary)
3. Save comprehensive report

## 🔒 Security Features

### API Key Protection
- **Encrypted Storage** using Fernet symmetric encryption
- **Secure Key Management** with restricted file permissions
- **API Key Validation** before storage and use

### Input Validation
- **Content Sanitization** removing dangerous patterns
- **Size Limits** preventing resource exhaustion
- **Format Validation** ensuring proper data types

### Rate Limiting
- **API Call Limits** preventing quota exhaustion
- **Failed Attempt Tracking** with automatic lockout
- **Session Management** with timeout protection

### Audit Logging
- **Security Event Logging** for monitoring
- **Access Tracking** for data operations
- **Error Logging** for troubleshooting

## 📊 Output Formats

### Executive Summary
- Key findings across all agents
- Overall confidence score
- Priority recommendations
- Implementation timeline

### Technical Report
- Detailed analysis from each agent
- Cross-domain synthesis
- Risk assessments
- Supporting evidence

### Individual Agent Reports
- Agent-specific findings
- Confidence levels
- Specialized recommendations
- Domain expertise insights

### Implementation Roadmap
- **Immediate (0-3 months)** - Critical actions
- **Short-term (3-12 months)** - Foundational work
- **Medium-term (1-3 years)** - Development phase
- **Long-term (3+ years)** - Strategic goals

## 🏗️ System Architecture

### Core Components

#### Multi-Agent System
```
EconomicPlanReviewManager
├── CentralPlanningAnalyst
├── LaborValueTheorist
├── MaterialConditionsExpert
├── SocialistDistributionSpecialist
├── ImplementationReviewer
├── WorkersDemocracyExpert
└── SocialDevelopmentAnalyst
```

#### Communication Hub
- Message routing between agents
- Consensus building mechanisms
- Conflict detection and resolution
- Cross-agent knowledge sharing

#### Security Layer
- Input validation and sanitization
- API key encryption and management
- Rate limiting and access control
- Audit logging and monitoring

#### Report Generation
- Multiple output formats
- Professional templates
- Custom formatting options
- Export capabilities

### Data Flow
1. **Input Validation** - Plan text sanitized and validated
2. **Agent Distribution** - Plan sent to selected agents
3. **Individual Analysis** - Each agent analyzes independently
4. **Cross-Communication** - Agents share findings
5. **Synthesis** - Central manager integrates results
6. **Report Generation** - Professional reports created
7. **Export** - Results saved in chosen format

## ⚙️ Configuration

### Security Settings
```python
SecurityConfig(
    max_plan_size=1_000_000,      # 1MB max
    max_api_calls_per_hour=100,   # Rate limit
    session_timeout=3600,         # 1 hour
    max_failed_attempts=5,        # Before lockout
    lockout_duration=1800         # 30 minutes
)
```

### Agent Selection
- **Recommended Set**: Core agents for general analysis
- **Full Set**: All 7 agents for comprehensive review
- **Custom**: Select specific agents based on needs

### Report Templates
- **Executive**: Concise summary for leadership
- **Technical**: Detailed analysis for economists
- **Policy**: Policy-focused brief for officials
- **Academic**: Scholarly report for research

## 🔧 Troubleshooting

### Common Issues

#### API Key Problems
- **Invalid Key**: Verify key from Google AI Studio
- **Expired Key**: Generate new key if needed
- **Rate Limits**: Wait or upgrade API plan
- **Network Issues**: Check internet connection

#### Performance Issues
- **Large Plans**: Break into smaller sections
- **Memory Usage**: Close other applications
- **Slow Analysis**: Reduce number of agents
- **Network Timeout**: Check connection stability

#### Security Warnings
- **File Permissions**: Ensure proper access rights
- **Encryption Errors**: Regenerate security keys
- **Validation Failures**: Check input format
- **Rate Limiting**: Wait before retry

### Error Recovery
The system includes automatic error recovery:
- **Exponential Backoff** for API retries
- **Graceful Degradation** if agents fail
- **Partial Results** when possible
- **Detailed Error Reporting** for debugging

## 📈 Performance Guidelines

### Optimal Usage
- **Plan Size**: 10,000-100,000 characters
- **Agent Count**: 3-5 agents for balance
- **Network**: Stable broadband connection
- **System**: 4GB+ RAM, modern CPU

### Resource Usage
- **Memory**: ~500MB-2GB depending on plan size
- **CPU**: Moderate during analysis phase
- **Network**: ~1-10MB per review session
- **Storage**: ~1MB per saved report

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create virtual environment
3. Install development dependencies
4. Run tests before submitting

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Include error handling

### Testing
```bash
pytest tests/ -v --cov=src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation
- **API Reference**: See individual module docstrings
- **User Guide**: This README file
- **Examples**: Check examples/ directory
- **FAQ**: Common questions and answers

### Getting Help
- **Issues**: Report bugs and feature requests
- **Discussions**: Community support and ideas
- **Documentation**: Comprehensive guides
- **Examples**: Sample economic plans and outputs

## 🔮 Future Enhancements

### Planned Features
- **Additional Agents**: More specialized domains
- **Custom Agents**: User-defined analysis focus
- **Batch Processing**: Multiple plans simultaneously
- **API Access**: REST API for integration
- **Visualization**: Charts and graphs
- **Collaboration**: Multi-user access

### Integration Options
- **Economic Modeling Software**: Input/output compatibility
- **Planning Systems**: Direct integration
- **Reporting Tools**: Export to BI platforms
- **Version Control**: Track plan revisions

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9+ | 3.11+ |
| RAM | 2GB | 4GB+ |
| Storage | 100MB | 1GB |
| Network | Broadband | High-speed |
| OS | Windows/Mac/Linux | Any modern OS |

## 🎯 Use Cases

### Economic Planning
- **National Plans**: Country-level economic strategies
- **Regional Development**: Local economic initiatives
- **Sector Analysis**: Industry-specific planning
- **Policy Evaluation**: Government policy assessment

### Academic Research
- **Economic Theory**: Theoretical framework analysis
- **Comparative Studies**: Cross-country comparisons
- **Historical Analysis**: Economic plan evolution
- **Methodology Development**: Planning technique research

### Policy Development
- **Government Planning**: Public sector strategies
- **Think Tank Analysis**: Policy research organizations
- **International Organizations**: Development planning
- **Consulting**: Professional economic advice

---

*Built with ❤️ for socialist economic planning and analysis*