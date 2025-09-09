#!/usr/bin/env python3
"""
Economic Plan Review System - Startup Script

Main entry point for the multi-agent economic plan review system.
Provides system initialization, health checks, and GUI launch.
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('economic_review_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tkinter',
        'google.generativeai',
        'cryptography',
        'PyPDF2',
        'docx',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'google.generativeai':
                import google.generativeai
            elif package == 'cryptography':
                import cryptography
            elif package == 'PyPDF2':
                import PyPDF2
            elif package == 'docx':
                import docx
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True


def check_system_requirements():
    """Check system requirements."""
    import platform
    import psutil
    
    logger.info("System Information:")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  Architecture: {platform.architecture()[0]}")
    
    # Check memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    logger.info(f"  Total RAM: {memory_gb:.1f} GB")
    
    if memory_gb < 2:
        logger.warning("⚠️  Low memory detected. Minimum 2GB recommended.")
    else:
        logger.info("✅ Sufficient memory available")
    
    # Check disk space
    disk = psutil.disk_usage('.')
    disk_gb_free = disk.free / (1024**3)
    logger.info(f"  Free disk space: {disk_gb_free:.1f} GB")
    
    if disk_gb_free < 1:
        logger.warning("⚠️  Low disk space. Minimum 1GB recommended.")
    else:
        logger.info("✅ Sufficient disk space available")
    
    return True


def initialize_directories():
    """Initialize required directories."""
    directories = [
        'logs',
        'exports',
        'cache',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Initialized directory: {directory}")
    
    return True


def run_system_tests():
    """Run basic system tests."""
    logger.info("Running system health checks...")
    
    try:
        # Test security components
        from cybernetic_planning.security.security_manager import SecurityManager, InputValidator
        
        security_manager = SecurityManager()
        validator = InputValidator()
        
        # Test input validation
        test_plan = "Test economic plan for validation. " * 10
        is_valid, message = validator.validate_economic_plan(test_plan)
        
        if is_valid:
            logger.info("✅ Input validation system working")
        else:
            logger.error(f"❌ Input validation failed: {message}")
            return False
        
        # Test communication hub
        from cybernetic_planning.agents.communication import CommunicationHub
        
        hub = CommunicationHub()
        hub.register_agent("test_agent", None)
        
        if "test_agent" in hub.agents:
            logger.info("✅ Communication system working")
        else:
            logger.error("❌ Communication system failed")
            return False
        
        # Test report generation
        from cybernetic_planning.agents.report_generator import ReportFormatter
        
        formatter = ReportFormatter()
        templates = formatter.get_available_templates()
        
        if len(templates) > 0:
            logger.info("✅ Report generation system working")
        else:
            logger.error("❌ Report generation system failed")
            return False
        
        logger.info("✅ All system health checks passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {str(e)}")
        return False


def launch_gui():
    """Launch the GUI application."""
    try:
        logger.info("🚀 Launching Economic Plan Review System GUI...")
        
        # Import and run GUI
        from economic_plan_review_gui import main
        main()
        
    except ImportError as e:
        logger.error(f"❌ Failed to import GUI module: {str(e)}")
        logger.error("Make sure economic_plan_review_gui.py is in the same directory")
        return False
    except Exception as e:
        logger.error(f"❌ GUI launch failed: {str(e)}")
        return False
    
    return True


def run_cli_mode():
    """Run in command-line interface mode."""
    print("\n" + "="*60)
    print("ECONOMIC PLAN REVIEW SYSTEM - CLI MODE")
    print("="*60)
    
    # Get API key
    api_key = input("Enter your Google Gemini 2.5 Pro API key: ").strip()
    
    if len(api_key) < 32:
        print("❌ API key too short. Please provide a valid key.")
        return False
    
    # Get plan input
    print("\nEnter your economic plan (press Ctrl+D or Ctrl+Z when finished):")
    plan_lines = []
    try:
        while True:
            line = input()
            plan_lines.append(line)
    except EOFError:
        pass
    
    economic_plan = '\n'.join(plan_lines).strip()
    
    if len(economic_plan) < 100:
        print("❌ Economic plan too short. Minimum 100 characters required.")
        return False
    
    # Select agents
    print("\nAvailable agents:")
    agents = [
        ("central_planning", "Central Planning Analyst"),
        ("labor_value", "Labor Value Theorist"),
        ("material_conditions", "Material Conditions Expert"),
        ("distribution", "Socialist Distribution Specialist"),
        ("implementation", "Implementation Reviewer"),
        ("democracy", "Workers' Democracy Expert"),
        ("social_development", "Social Development Analyst")
    ]
    
    for i, (agent_id, name) in enumerate(agents, 1):
        print(f"  {i}. {name}")
    
    selected_indices = input("\nSelect agents (e.g., 1,3,5 or 'all'): ").strip()
    
    if selected_indices.lower() == 'all':
        selected_agents = [agent_id for agent_id, _ in agents]
    else:
        try:
            indices = [int(i.strip()) - 1 for i in selected_indices.split(',')]
            selected_agents = [agents[i][0] for i in indices if 0 <= i < len(agents)]
        except (ValueError, IndexError):
            print("❌ Invalid agent selection.")
            return False
    
    if not selected_agents:
        print("❌ No agents selected.")
        return False
    
    # Run review
    try:
        print(f"\n🔄 Starting review with {len(selected_agents)} agents...")
        print("This may take several minutes...")
        
        from cybernetic_planning.agents.review_manager import EconomicPlanReviewManager
        
        manager = EconomicPlanReviewManager(api_key)
        session_id = manager.start_review_session(economic_plan)
        
        comprehensive_review = manager.conduct_review(session_id, selected_agents)
        
        # Display results
        print("\n" + "="*60)
        print("REVIEW RESULTS")
        print("="*60)
        
        print(f"\nSession ID: {comprehensive_review.session_id}")
        print(f"Confidence Score: {comprehensive_review.confidence_score:.2f}/1.0")
        print(f"Agents Used: {len(selected_agents)}")
        
        print(f"\nINTEGRATED SUMMARY:")
        print("-" * 20)
        print(comprehensive_review.integrated_summary)
        
        print(f"\nOVERALL ASSESSMENT:")
        print("-" * 20)
        print(comprehensive_review.overall_assessment)
        
        print(f"\nTOP RECOMMENDATIONS:")
        print("-" * 20)
        for i, rec in enumerate(comprehensive_review.prioritized_recommendations[:5], 1):
            print(f"{i}. {rec['recommendation']}")
            print(f"   (by {rec['agent_name']}, confidence: {rec['agent_confidence']:.2f})")
        
        # Save report
        save_report = input("\nSave detailed report to file? (y/n): ").strip().lower()
        if save_report == 'y':
            filename = f"economic_review_{session_id}.txt"
            
            from cybernetic_planning.agents.report_generator import create_detailed_report
            session = manager.active_sessions[session_id]
            report_content = create_detailed_report(comprehensive_review, session)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"✅ Report saved to: {filename}")
        
        print(f"\n✅ Review completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Review failed: {str(e)}")
        logger.error(f"CLI review failed: {str(e)}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Economic Plan Review System - Multi-Agent AI Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_economic_review_system.py              # Launch GUI (default)
  python run_economic_review_system.py --cli        # Run in CLI mode
  python run_economic_review_system.py --test       # Run system tests only
  python run_economic_review_system.py --check      # Check system health
        """
    )
    
    parser.add_argument('--cli', action='store_true', 
                       help='Run in command-line interface mode')
    parser.add_argument('--test', action='store_true', 
                       help='Run system tests only')
    parser.add_argument('--check', action='store_true', 
                       help='Check system health and requirements')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip system health checks')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*60)
    print("ECONOMIC PLAN REVIEW SYSTEM")
    print("Multi-Agent AI Analysis for Socialist Economic Planning")
    print("="*60)
    print("Version: 1.0.0")
    print("Powered by Google Gemini 2.5 Pro API")
    print("="*60)
    
    # Initialize system
    logger.info("Initializing Economic Plan Review System...")
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check system requirements
        if not check_system_requirements():
            sys.exit(1)
        
        # Initialize directories
        if not initialize_directories():
            sys.exit(1)
    
    # Run based on mode
    if args.check:
        logger.info("Running system health checks...")
        if run_system_tests():
            print("✅ System is healthy and ready to use!")
            sys.exit(0)
        else:
            print("❌ System health check failed!")
            sys.exit(1)
    
    elif args.test:
        logger.info("Running comprehensive system tests...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, 'test_economic_review_system.py'], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            sys.exit(result.returncode)
        except Exception as e:
            logger.error(f"Failed to run tests: {str(e)}")
            sys.exit(1)
    
    elif args.cli:
        logger.info("Starting CLI mode...")
        if not args.skip_checks:
            if not run_system_tests():
                sys.exit(1)
        
        if run_cli_mode():
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        # Default: GUI mode
        logger.info("Starting GUI mode...")
        if not args.skip_checks:
            if not run_system_tests():
                print("❌ System health check failed. Use --skip-checks to bypass.")
                sys.exit(1)
        
        if launch_gui():
            logger.info("GUI session completed")
        else:
            logger.error("GUI launch failed")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  System interrupted by user")
        logger.info("System interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        logger.error(f"System error: {str(e)}", exc_info=True)
        sys.exit(1)