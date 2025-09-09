#!/usr/bin/env python3
"""
Economic Plan Review System GUI

A comprehensive GUI application for multi-agent economic plan review using
Google Gemini 2.5 Pro API with specialized AI agents for socialist economic analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
import PyPDF2
import docx
from cryptography.fernet import Fernet
import base64

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cybernetic_planning.agents.review_manager import EconomicPlanReviewManager
    from cybernetic_planning.agents.communication import CommunicationHub
except ImportError as e:
    print(f"Error importing review system: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class SecureAPIKeyManager:
    """Secure API key management with encryption."""
    
    def __init__(self, config_file: str = ".api_keys.enc"):
        self.config_file = config_file
        self.key_file = ".key"
        self._fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption key."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        
        self._fernet = Fernet(key)
    
    def save_api_key(self, key_name: str, api_key: str):
        """Save encrypted API key."""
        config = self.load_config()
        config[key_name] = self._fernet.encrypt(api_key.encode()).decode()
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
    
    def load_api_key(self, key_name: str) -> str:
        """Load decrypted API key."""
        config = self.load_config()
        if key_name in config:
            encrypted_key = config[key_name].encode()
            return self._fernet.decrypt(encrypted_key).decode()
        return ""
    
    def load_config(self) -> dict:
        """Load configuration file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def test_api_key(self, api_key: str) -> bool:
        """Test if API key is valid."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content("Test connection")
            return True
        except Exception as e:
            print(f"API key test failed: {e}")
            return False


class DocumentProcessor:
    """Process various document formats for plan input."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Failed to read TXT: {str(e)}")
    
    @staticmethod
    def process_document(file_path: str) -> str:
        """Process document and extract text."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return DocumentProcessor.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return DocumentProcessor.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return DocumentProcessor.extract_text_from_txt(file_path)
        else:
            raise Exception(f"Unsupported file format: {file_ext}")


class EconomicPlanReviewGUI:
    """Main GUI class for the Economic Plan Review System."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Economic Plan Review System - Multi-Agent AI Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Initialize components
        self.api_key_manager = SecureAPIKeyManager()
        self.review_manager = None
        self.communication_hub = CommunicationHub()
        self.current_session = None
        self.current_review = None
        
        # GUI state
        self.api_key_valid = False
        self.plan_text = ""
        self.selected_agents = []
        
        # Create GUI elements
        self.create_widgets()
        self.setup_layout()
        
        # Initialize API key check
        self.check_api_key_on_startup()
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Create tabs
        self.create_api_key_tab()
        self.create_plan_input_tab()
        self.create_agent_selection_tab()
        self.create_review_progress_tab()
        self.create_results_tab()
        self.create_export_tab()
        self.create_about_tab()
    
    def create_api_key_tab(self):
        """Create API key management tab."""
        self.api_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.api_frame, text="🔑 API Configuration")
        
        # API Key Status
        status_frame = ttk.LabelFrame(self.api_frame, text="API Key Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.api_status_label = ttk.Label(status_frame, text="🔴 API Key Not Configured", 
                                         font=("Arial", 12, "bold"), foreground="red")
        self.api_status_label.pack(pady=5)
        
        # API Key Input
        input_frame = ttk.LabelFrame(self.api_frame, text="Google Gemini 2.5 Pro API Key", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="Enter your Google Gemini API Key:").pack(anchor="w", pady=(0, 5))
        
        key_entry_frame = ttk.Frame(input_frame)
        key_entry_frame.pack(fill="x", pady=5)
        
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(key_entry_frame, textvariable=self.api_key_var, 
                                      show="*", width=50, font=("Courier", 10))
        self.api_key_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.show_key_var = tk.BooleanVar()
        show_key_cb = ttk.Checkbutton(key_entry_frame, text="Show", variable=self.show_key_var,
                                     command=self.toggle_key_visibility)
        show_key_cb.pack(side="right")
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="Test API Key", command=self.test_api_key,
                  style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save API Key", command=self.save_api_key).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load Saved Key", command=self.load_saved_key).pack(side="left", padx=5)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(self.api_frame, text="Setup Instructions", padding=10)
        instructions_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        instructions_text = scrolledtext.ScrolledText(instructions_frame, height=15, wrap=tk.WORD)
        instructions_text.pack(fill="both", expand=True)
        
        instructions_content = """
HOW TO GET GOOGLE GEMINI 2.5 PRO API KEY:

1. Visit Google AI Studio:
   • Go to https://aistudio.google.com/
   • Sign in with your Google account

2. Create API Key:
   • Click on "Get API key" or "Create API key"
   • Select "Create API key in new project" or choose existing project
   • Copy the generated API key

3. API Key Security:
   • Keep your API key secure and private
   • Do not share it publicly or commit to version control
   • The key will be encrypted and stored locally

4. Usage Limits:
   • Free tier includes generous usage limits
   • Monitor your usage in Google AI Studio
   • Consider upgrading for production use

5. Troubleshooting:
   • Ensure the API key is active and not expired
   • Check that Gemini API is enabled for your project
   • Verify network connectivity

SYSTEM REQUIREMENTS:
• Python 3.9 or higher
• Internet connection for API calls
• Sufficient memory for processing large economic plans

PRIVACY & SECURITY:
• API keys are encrypted using Fernet symmetric encryption
• Economic plan data is processed securely through Google's API
• No data is stored on external servers beyond API processing
• All analysis results are stored locally

The Economic Plan Review System uses multiple specialized AI agents:
• Central Planning Analyst - Production planning and resource allocation
• Labor Value Theorist - Labor theory of value and productivity analysis
• Material Conditions Expert - Material dialectics and productive forces
• Socialist Distribution Specialist - Distribution mechanisms and social needs
• Implementation Reviewer - Feasibility and coordination analysis
• Workers' Democracy Expert - Democratic participation and worker control
• Social Development Analyst - Social development and class analysis

Each agent provides professional analysis from their specialized perspective,
contributing to a comprehensive multi-dimensional review of economic plans.
        """
        
        instructions_text.insert("1.0", instructions_content)
        instructions_text.config(state=tk.DISABLED)
    
    def create_plan_input_tab(self):
        """Create plan input tab."""
        self.plan_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plan_frame, text="📄 Plan Input")
        
        # File input section
        file_frame = ttk.LabelFrame(self.plan_frame, text="Load Economic Plan", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        file_button_frame = ttk.Frame(file_frame)
        file_button_frame.pack(fill="x", pady=5)
        
        ttk.Button(file_button_frame, text="📁 Load from File", 
                  command=self.load_plan_from_file).pack(side="left", padx=5)
        ttk.Button(file_button_frame, text="📋 Paste from Clipboard", 
                  command=self.paste_from_clipboard).pack(side="left", padx=5)
        ttk.Button(file_button_frame, text="🗑️ Clear", 
                  command=self.clear_plan_text).pack(side="left", padx=5)
        
        self.file_info_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.file_info_label.pack(anchor="w", pady=5)
        
        # Text input section
        text_frame = ttk.LabelFrame(self.plan_frame, text="Economic Plan Text", padding=10)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.plan_text_widget = scrolledtext.ScrolledText(text_frame, height=20, wrap=tk.WORD,
                                                         font=("Consolas", 10))
        self.plan_text_widget.pack(fill="both", expand=True)
        
        # Character count
        self.char_count_label = ttk.Label(text_frame, text="Characters: 0")
        self.char_count_label.pack(anchor="w", pady=5)
        
        # Bind text change event
        self.plan_text_widget.bind('<KeyRelease>', self.update_char_count)
    
    def create_agent_selection_tab(self):
        """Create agent selection tab."""
        self.agent_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.agent_frame, text="🤖 Agent Selection")
        
        # Agent selection
        selection_frame = ttk.LabelFrame(self.agent_frame, text="Select Analysis Agents", padding=10)
        selection_frame.pack(fill="x", padx=10, pady=5)
        
        # Select all/none buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="Select All", command=self.select_all_agents).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Select None", command=self.select_no_agents).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Recommended Set", command=self.select_recommended_agents).pack(side="left", padx=5)
        
        # Agent checkboxes
        self.agent_vars = {}
        agents_info = [
            ("central_planning", "Central Planning Analyst", "Production planning, resource allocation, output targets"),
            ("labor_value", "Labor Value Theorist", "Labor theory of value, surplus value analysis, productivity"),
            ("material_conditions", "Material Conditions Expert", "Material dialectics, productive forces, relations of production"),
            ("distribution", "Socialist Distribution Specialist", "Distribution mechanisms, social needs fulfillment"),
            ("implementation", "Implementation Reviewer", "Feasibility, timeline, resource coordination"),
            ("democracy", "Workers' Democracy Expert", "Democratic participation, worker control, collective decision-making"),
            ("social_development", "Social Development Analyst", "Social development, class analysis, exploitation elimination")
        ]
        
        for agent_id, name, description in agents_info:
            var = tk.BooleanVar(value=True)  # Default to selected
            self.agent_vars[agent_id] = var
            
            agent_frame = ttk.Frame(selection_frame)
            agent_frame.pack(fill="x", pady=2)
            
            cb = ttk.Checkbutton(agent_frame, text=name, variable=var, 
                               command=self.update_selected_agents)
            cb.pack(side="left")
            
            desc_label = ttk.Label(agent_frame, text=f"  —  {description}", 
                                 foreground="gray", font=("Arial", 9))
            desc_label.pack(side="left")
        
        # Agent details
        details_frame = ttk.LabelFrame(self.agent_frame, text="Agent Details", padding=10)
        details_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.agent_details_text = scrolledtext.ScrolledText(details_frame, height=15, wrap=tk.WORD)
        self.agent_details_text.pack(fill="both", expand=True)
        
        # Load agent details
        self.load_agent_details()
        
        # Update selected agents
        self.update_selected_agents()
    
    def create_review_progress_tab(self):
        """Create review progress tab."""
        self.progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_frame, text="⚙️ Review Progress")
        
        # Review controls
        control_frame = ttk.LabelFrame(self.progress_frame, text="Review Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_review_button = ttk.Button(control_frame, text="🚀 Start Review", 
                                            command=self.start_review, style="Accent.TButton",
                                            state="disabled")
        self.start_review_button.pack(side="left", padx=5)
        
        self.stop_review_button = ttk.Button(control_frame, text="⏹️ Stop Review", 
                                           command=self.stop_review, state="disabled")
        self.stop_review_button.pack(side="left", padx=5)
        
        # Progress indicators
        progress_frame = ttk.LabelFrame(self.progress_frame, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress.pack(fill="x", pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to start review")
        self.progress_label.pack(anchor="w")
        
        # Agent status
        status_frame = ttk.LabelFrame(self.progress_frame, text="Agent Status", padding=10)
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for agent status
        columns = ("agent", "status", "confidence", "completion")
        self.status_tree = ttk.Treeview(status_frame, columns=columns, show="headings", height=10)
        
        self.status_tree.heading("agent", text="Agent")
        self.status_tree.heading("status", text="Status")
        self.status_tree.heading("confidence", text="Confidence")
        self.status_tree.heading("completion", text="Completion Time")
        
        self.status_tree.column("agent", width=200)
        self.status_tree.column("status", width=100)
        self.status_tree.column("confidence", width=100)
        self.status_tree.column("completion", width=150)
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_tree.yview)
        self.status_tree.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_tree.pack(side="left", fill="both", expand=True)
        status_scrollbar.pack(side="right", fill="y")
    
    def create_results_tab(self):
        """Create results display tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="📊 Results")
        
        # Results notebook for different views
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Executive Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, wrap=tk.WORD, 
                                                     font=("Arial", 11))
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Individual reports tab
        self.individual_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.individual_frame, text="Individual Agent Reports")
        
        # Agent selector for individual reports
        agent_select_frame = ttk.Frame(self.individual_frame)
        agent_select_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(agent_select_frame, text="Select Agent:").pack(side="left", padx=5)
        self.agent_report_var = tk.StringVar()
        self.agent_report_combo = ttk.Combobox(agent_select_frame, textvariable=self.agent_report_var,
                                              state="readonly", width=30)
        self.agent_report_combo.pack(side="left", padx=5)
        self.agent_report_combo.bind("<<ComboboxSelected>>", self.show_individual_report)
        
        self.individual_text = scrolledtext.ScrolledText(self.individual_frame, wrap=tk.WORD,
                                                        font=("Arial", 10))
        self.individual_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Recommendations tab
        self.recommendations_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.recommendations_frame, text="Recommendations")
        
        self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_frame, wrap=tk.WORD,
                                                            font=("Arial", 10))
        self.recommendations_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Implementation roadmap tab
        self.roadmap_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.roadmap_frame, text="Implementation Roadmap")
        
        self.roadmap_text = scrolledtext.ScrolledText(self.roadmap_frame, wrap=tk.WORD,
                                                     font=("Arial", 10))
        self.roadmap_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_export_tab(self):
        """Create export tab."""
        self.export_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.export_frame, text="💾 Export")
        
        # Export options
        options_frame = ttk.LabelFrame(self.export_frame, text="Export Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(options_frame, text="📄 Export as Text Report", 
                  command=lambda: self.export_results("text")).pack(side="left", padx=5)
        ttk.Button(options_frame, text="📋 Export as JSON", 
                  command=lambda: self.export_results("json")).pack(side="left", padx=5)
        ttk.Button(options_frame, text="📊 Export Summary Only", 
                  command=lambda: self.export_results("summary")).pack(side="left", padx=5)
        
        # Export status
        self.export_status_label = ttk.Label(self.export_frame, text="No review to export", 
                                           foreground="gray")
        self.export_status_label.pack(pady=10)
        
        # Export preview
        preview_frame = ttk.LabelFrame(self.export_frame, text="Export Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.export_preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD,
                                                           font=("Consolas", 9))
        self.export_preview_text.pack(fill="both", expand=True)
    
    def create_about_tab(self):
        """Create about tab."""
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="ℹ️ About")
        
        about_text = scrolledtext.ScrolledText(self.about_frame, wrap=tk.WORD, 
                                              font=("Arial", 11), state=tk.DISABLED)
        about_text.pack(fill="both", expand=True, padx=20, pady=20)
        
        about_content = """
ECONOMIC PLAN REVIEW SYSTEM
Multi-Agent AI Analysis for Socialist Economic Planning

VERSION: 1.0.0
DEVELOPED: 2024

OVERVIEW:
This system provides comprehensive review and analysis of economic plans through 
specialized AI agents working collaboratively. Each agent brings expertise in 
different aspects of socialist economic theory and planning.

SPECIALIZED AGENTS:

🏭 Central Planning Analyst
• Production planning and capacity analysis
• Resource allocation optimization  
• Output targets and feasibility assessment
• Sectoral coordination and material balance

⚒️ Labor Value Theorist
• Labor theory of value applications
• Surplus value analysis and distribution
• Productivity measurement and improvement
• Socially necessary labor time calculations

🔬 Material Conditions Expert
• Material dialectics and historical materialism
• Productive forces development
• Relations of production analysis
• Infrastructure and technological assessment

🤝 Socialist Distribution Specialist
• "From each according to ability, to each according to need"
• Social needs assessment and prioritization
• Public goods provision and accessibility
• Universal basic services implementation

⚙️ Implementation Reviewer
• Implementation feasibility assessment
• Timeline analysis and milestone planning
• Resource coordination and logistics
• Risk management and contingency planning

🗳️ Workers' Democracy Expert
• Democratic participation mechanisms
• Worker control and workplace democracy
• Collective decision-making processes
• Participatory budgeting and resource allocation

🌱 Social Development Analyst
• Social development and human welfare
• Class structure and social stratification
• Exploitation elimination measures
• Education, healthcare, and social services

FEATURES:
• Multi-agent collaborative analysis
• Professional, academic-quality reports
• Cross-domain synthesis and integration
• Conflict detection and resolution
• Implementation roadmap generation
• Secure API key management
• Support for multiple document formats
• Comprehensive export options

TECHNOLOGY:
• Google Gemini 2.5 Pro API for AI analysis
• Python with Tkinter GUI framework
• Encrypted API key storage
• Multi-threaded processing
• Professional report generation

USAGE:
1. Configure your Google Gemini API key
2. Load or paste your economic plan
3. Select specialized agents for analysis
4. Start the multi-agent review process
5. Review comprehensive results and recommendations
6. Export detailed reports for implementation

This system is designed for economists, planners, policy makers, and researchers 
working on socialist economic planning and analysis.

For support and updates, visit the project repository.
        """
        
        about_text.config(state=tk.NORMAL)
        about_text.insert("1.0", about_content)
        about_text.config(state=tk.DISABLED)
    
    def setup_layout(self):
        """Setup the main layout."""
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure styles
        style = ttk.Style()
        style.configure("Accent.TButton", foreground="white", background="#0078d4")
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def check_api_key_on_startup(self):
        """Check for saved API key on startup."""
        saved_key = self.api_key_manager.load_api_key("gemini_api_key")
        if saved_key:
            self.api_key_var.set(saved_key)
            self.test_api_key()
    
    def toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
    
    def test_api_key(self):
        """Test the API key validity."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first")
            return
        
        self.status_bar.config(text="Testing API key...")
        self.root.update()
        
        try:
            if self.api_key_manager.test_api_key(api_key):
                self.api_key_valid = True
                self.api_status_label.config(text="🟢 API Key Valid", foreground="green")
                self.start_review_button.config(state="normal")
                self.status_bar.config(text="API key validated successfully")
                messagebox.showinfo("Success", "API key is valid and working!")
            else:
                self.api_key_valid = False
                self.api_status_label.config(text="🔴 API Key Invalid", foreground="red")
                self.start_review_button.config(state="disabled")
                self.status_bar.config(text="API key validation failed")
                messagebox.showerror("Error", "API key is invalid or not working")
        except Exception as e:
            self.api_key_valid = False
            self.api_status_label.config(text="🔴 API Key Test Failed", foreground="red")
            self.start_review_button.config(state="disabled")
            self.status_bar.config(text="API key test error")
            messagebox.showerror("Error", f"Failed to test API key: {str(e)}")
    
    def save_api_key(self):
        """Save the API key securely."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key first")
            return
        
        try:
            self.api_key_manager.save_api_key("gemini_api_key", api_key)
            messagebox.showinfo("Success", "API key saved securely")
            self.status_bar.config(text="API key saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save API key: {str(e)}")
    
    def load_saved_key(self):
        """Load previously saved API key."""
        try:
            saved_key = self.api_key_manager.load_api_key("gemini_api_key")
            if saved_key:
                self.api_key_var.set(saved_key)
                self.status_bar.config(text="API key loaded from secure storage")
                messagebox.showinfo("Success", "API key loaded successfully")
            else:
                messagebox.showinfo("Info", "No saved API key found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load API key: {str(e)}")
    
    def load_plan_from_file(self):
        """Load economic plan from file."""
        file_path = filedialog.askopenfilename(
            title="Select Economic Plan Document",
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_bar.config(text="Loading document...")
                self.root.update()
                
                text = DocumentProcessor.process_document(file_path)
                self.plan_text_widget.delete("1.0", tk.END)
                self.plan_text_widget.insert("1.0", text)
                
                file_name = os.path.basename(file_path)
                self.file_info_label.config(text=f"Loaded: {file_name}", foreground="green")
                self.status_bar.config(text=f"Document loaded: {file_name}")
                self.update_char_count()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load document: {str(e)}")
                self.status_bar.config(text="Failed to load document")
    
    def paste_from_clipboard(self):
        """Paste text from clipboard."""
        try:
            clipboard_text = self.root.clipboard_get()
            self.plan_text_widget.delete("1.0", tk.END)
            self.plan_text_widget.insert("1.0", clipboard_text)
            self.file_info_label.config(text="Pasted from clipboard", foreground="blue")
            self.update_char_count()
            self.status_bar.config(text="Text pasted from clipboard")
        except tk.TclError:
            messagebox.showwarning("Warning", "No text found in clipboard")
    
    def clear_plan_text(self):
        """Clear plan text."""
        self.plan_text_widget.delete("1.0", tk.END)
        self.file_info_label.config(text="No file loaded", foreground="gray")
        self.update_char_count()
        self.status_bar.config(text="Plan text cleared")
    
    def update_char_count(self, event=None):
        """Update character count display."""
        text = self.plan_text_widget.get("1.0", tk.END)
        char_count = len(text) - 1  # Subtract 1 for the extra newline
        self.char_count_label.config(text=f"Characters: {char_count:,}")
    
    def select_all_agents(self):
        """Select all agents."""
        for var in self.agent_vars.values():
            var.set(True)
        self.update_selected_agents()
    
    def select_no_agents(self):
        """Select no agents."""
        for var in self.agent_vars.values():
            var.set(False)
        self.update_selected_agents()
    
    def select_recommended_agents(self):
        """Select recommended set of agents."""
        recommended = ["central_planning", "distribution", "implementation", "social_development"]
        for agent_id, var in self.agent_vars.items():
            var.set(agent_id in recommended)
        self.update_selected_agents()
    
    def update_selected_agents(self):
        """Update list of selected agents."""
        self.selected_agents = [agent_id for agent_id, var in self.agent_vars.items() if var.get()]
        count = len(self.selected_agents)
        self.status_bar.config(text=f"Selected {count} agents for analysis")
    
    def load_agent_details(self):
        """Load detailed information about agents."""
        details = """
SPECIALIZED ECONOMIC REVIEW AGENTS

Each agent brings deep expertise in their domain and analyzes economic plans from 
their specialized perspective, contributing to a comprehensive multi-dimensional review.

🏭 CENTRAL PLANNING ANALYST
Specialization: Central Planning and Resource Allocation
• Evaluates production targets for realism and achievability
• Assesses resource allocation efficiency across sectors
• Identifies potential bottlenecks or coordination issues
• Analyzes sectoral interdependencies and material flows
• Reviews capacity constraints and expansion requirements
• Examines plan coordination mechanisms

⚒️ LABOR VALUE THEORIST  
Specialization: Labor Theory of Value and Productivity Analysis
• Evaluates labor value calculations and methodologies
• Assesses surplus value creation and distribution patterns
• Analyzes productivity trends and improvement potential
• Reviews labor allocation efficiency across sectors
• Identifies opportunities for socially necessary labor time reduction
• Examines skill development and training requirements

🔬 MATERIAL CONDITIONS EXPERT
Specialization: Material Dialectics and Productive Forces
• Analyzes the material foundation of economic plans
• Evaluates productive forces development trajectories
• Assesses technology and infrastructure requirements
• Reviews environmental sustainability considerations
• Examines resource constraints and availability
• Identifies contradictions between productive forces and relations

🤝 SOCIALIST DISTRIBUTION SPECIALIST
Specialization: Socialist Distribution and Social Needs
• Evaluates distribution mechanisms for fairness and efficiency
• Assesses social needs coverage and prioritization systems
• Analyzes public goods and services provision strategies
• Reviews accessibility and universal access mechanisms
• Identifies gaps in social needs fulfillment
• Examines community participation in distribution decisions

⚙️ IMPLEMENTATION REVIEWER
Specialization: Implementation Feasibility and Coordination
• Evaluates implementation feasibility and realistic timelines
• Assesses resource coordination requirements
• Analyzes institutional capacity and organizational readiness
• Identifies implementation risks and mitigation strategies
• Reviews monitoring and evaluation frameworks
• Examines administrative and organizational structures needed

🗳️ WORKERS' DEMOCRACY EXPERT
Specialization: Workers' Democracy and Collective Decision-Making
• Evaluates democratic participation mechanisms in planning
• Assesses worker control and decision-making power structures
• Analyzes collective decision-making processes effectiveness
• Reviews community involvement and representation systems
• Identifies opportunities for increased democratic participation
• Examines governance structures for democratic accountability

🌱 SOCIAL DEVELOPMENT ANALYST
Specialization: Social Development and Class Analysis
• Evaluates social development outcomes and target achievement
• Analyzes class relations and power structure transformations
• Assesses exploitation elimination measures effectiveness
• Reviews social services and welfare provision systems
• Identifies opportunities for progressive social change
• Examines community development and empowerment initiatives

COLLABORATIVE ANALYSIS PROCESS:
1. Individual Analysis - Each agent analyzes the plan independently
2. Cross-Agent Communication - Agents share findings and identify synergies
3. Conflict Resolution - Disagreements are identified and resolved
4. Synthesis - All findings are integrated into comprehensive review
5. Final Report - Unified assessment with prioritized recommendations
        """
        
        self.agent_details_text.delete("1.0", tk.END)
        self.agent_details_text.insert("1.0", details)
    
    def start_review(self):
        """Start the multi-agent review process."""
        # Validation
        if not self.api_key_valid:
            messagebox.showerror("Error", "Please configure and validate your API key first")
            return
        
        plan_text = self.plan_text_widget.get("1.0", tk.END).strip()
        if not plan_text:
            messagebox.showerror("Error", "Please enter an economic plan to review")
            return
        
        if not self.selected_agents:
            messagebox.showerror("Error", "Please select at least one agent for analysis")
            return
        
        # Confirm start
        agent_names = [name for agent_id, name, _ in [
            ("central_planning", "Central Planning Analyst", ""),
            ("labor_value", "Labor Value Theorist", ""),
            ("material_conditions", "Material Conditions Expert", ""),
            ("distribution", "Socialist Distribution Specialist", ""),
            ("implementation", "Implementation Reviewer", ""),
            ("democracy", "Workers' Democracy Expert", ""),
            ("social_development", "Social Development Analyst", "")
        ] if agent_id in self.selected_agents]
        
        confirm_msg = f"""Start multi-agent economic plan review?

Selected Agents ({len(self.selected_agents)}):
{chr(10).join(f'• {name}' for name in agent_names)}

Plan Length: {len(plan_text):,} characters

This process may take several minutes depending on plan complexity.
Continue?"""
        
        if not messagebox.askyesno("Confirm Review", confirm_msg):
            return
        
        # Initialize review manager
        try:
            self.review_manager = EconomicPlanReviewManager(self.api_key_var.get().strip())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize review system: {str(e)}")
            return
        
        # Update UI state
        self.start_review_button.config(state="disabled")
        self.stop_review_button.config(state="normal")
        self.progress_label.config(text="Starting multi-agent review...")
        self.overall_progress.config(mode='indeterminate')
        self.overall_progress.start()
        
        # Clear previous results
        self.clear_results()
        
        # Populate agent status tree
        self.populate_agent_status_tree()
        
        # Start review in separate thread
        self.review_thread = threading.Thread(target=self.run_review, args=(plan_text,))
        self.review_thread.daemon = True
        self.review_thread.start()
        
        # Switch to progress tab
        self.notebook.select(self.progress_frame)
    
    def run_review(self, plan_text):
        """Run the review process in a separate thread."""
        try:
            # Start review session
            self.current_session = self.review_manager.start_review_session(plan_text)
            
            # Update progress
            self.root.after(0, lambda: self.progress_label.config(text="Conducting multi-agent analysis..."))
            
            # Conduct review
            self.current_review = self.review_manager.conduct_review(
                self.current_session, 
                self.selected_agents
            )
            
            # Review completed successfully
            self.root.after(0, self.review_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.review_failed(str(e)))
    
    def review_completed(self):
        """Handle successful review completion."""
        self.start_review_button.config(state="normal")
        self.stop_review_button.config(state="disabled")
        self.overall_progress.stop()
        self.overall_progress.config(mode='determinate', value=100)
        self.progress_label.config(text="✅ Review completed successfully!")
        
        # Update agent status
        self.update_agent_status_tree()
        
        # Display results
        self.display_results()
        
        # Update export status
        self.export_status_label.config(text="✅ Review ready for export", foreground="green")
        
        # Switch to results tab
        self.notebook.select(self.results_frame)
        
        self.status_bar.config(text="Multi-agent review completed successfully")
        
        # Show completion message
        messagebox.showinfo("Review Complete", 
                          f"Multi-agent economic plan review completed!\n\n"
                          f"Agents: {len(self.selected_agents)}\n"
                          f"Session: {self.current_session}\n"
                          f"Confidence: {self.current_review.confidence_score:.2f}")
    
    def review_failed(self, error_message):
        """Handle review failure."""
        self.start_review_button.config(state="normal")
        self.stop_review_button.config(state="disabled")
        self.overall_progress.stop()
        self.overall_progress.config(value=0)
        self.progress_label.config(text="❌ Review failed")
        
        self.status_bar.config(text="Review failed")
        messagebox.showerror("Review Failed", f"The review process failed:\n\n{error_message}")
    
    def stop_review(self):
        """Stop the review process."""
        if messagebox.askyesno("Stop Review", "Are you sure you want to stop the review process?"):
            # Note: In a full implementation, you would need to implement proper thread cancellation
            self.start_review_button.config(state="normal")
            self.stop_review_button.config(state="disabled")
            self.overall_progress.stop()
            self.overall_progress.config(value=0)
            self.progress_label.config(text="Review stopped by user")
            self.status_bar.config(text="Review stopped")
    
    def populate_agent_status_tree(self):
        """Populate the agent status tree."""
        # Clear existing items
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        
        # Add selected agents
        agent_names = {
            "central_planning": "Central Planning Analyst",
            "labor_value": "Labor Value Theorist", 
            "material_conditions": "Material Conditions Expert",
            "distribution": "Socialist Distribution Specialist",
            "implementation": "Implementation Reviewer",
            "democracy": "Workers' Democracy Expert",
            "social_development": "Social Development Analyst"
        }
        
        for agent_id in self.selected_agents:
            name = agent_names.get(agent_id, agent_id)
            self.status_tree.insert("", "end", values=(name, "⏳ Analyzing...", "—", "—"))
    
    def update_agent_status_tree(self):
        """Update agent status after review completion."""
        if not self.current_review:
            return
        
        # Get session data
        session = self.review_manager.active_sessions.get(self.current_session)
        if not session:
            return
        
        # Clear and repopulate with results
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
        
        agent_names = {
            "central_planning": "Central Planning Analyst",
            "labor_value": "Labor Value Theorist",
            "material_conditions": "Material Conditions Expert", 
            "distribution": "Socialist Distribution Specialist",
            "implementation": "Implementation Reviewer",
            "democracy": "Workers' Democracy Expert",
            "social_development": "Social Development Analyst"
        }
        
        for agent_id in self.selected_agents:
            name = agent_names.get(agent_id, agent_id)
            
            if agent_id in session.agent_reports:
                report = session.agent_reports[agent_id]
                confidence = f"{report.confidence_level:.2f}"
                completion_time = time.strftime('%H:%M:%S', time.localtime(report.timestamp))
                status = "✅ Complete"
            else:
                confidence = "—"
                completion_time = "—"
                status = "❌ Failed"
            
            self.status_tree.insert("", "end", values=(name, status, confidence, completion_time))
    
    def clear_results(self):
        """Clear previous results."""
        self.summary_text.delete("1.0", tk.END)
        self.individual_text.delete("1.0", tk.END)
        self.recommendations_text.delete("1.0", tk.END)
        self.roadmap_text.delete("1.0", tk.END)
        self.export_preview_text.delete("1.0", tk.END)
        
        # Clear agent report combo
        self.agent_report_combo['values'] = []
        self.agent_report_var.set("")
    
    def display_results(self):
        """Display review results."""
        if not self.current_review:
            return
        
        # Display integrated summary
        summary = f"""ECONOMIC PLAN REVIEW - INTEGRATED SUMMARY
{'='*60}

Session ID: {self.current_review.session_id}
Review Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.current_review.timestamp))}
Overall Confidence Score: {self.current_review.confidence_score:.2f}/1.0
Agents Participating: {len(self.selected_agents)}

INTEGRATED SUMMARY:
{self.current_review.integrated_summary}

CROSS-DOMAIN ANALYSIS:
{self.current_review.cross_domain_analysis}

OVERALL ASSESSMENT:
{self.current_review.overall_assessment}
"""
        
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary)
        
        # Display recommendations
        recommendations = "PRIORITIZED RECOMMENDATIONS\n" + "="*40 + "\n\n"
        
        for i, rec in enumerate(self.current_review.prioritized_recommendations, 1):
            recommendations += f"{i:2d}. {rec['recommendation']}\n"
            recommendations += f"    Agent: {rec['agent_name']}\n"
            recommendations += f"    Confidence: {rec['agent_confidence']:.2f}\n"
            recommendations += f"    Priority Score: {rec['priority_score']:.2f}\n\n"
        
        self.recommendations_text.delete("1.0", tk.END)
        self.recommendations_text.insert("1.0", recommendations)
        
        # Display implementation roadmap
        roadmap = "IMPLEMENTATION ROADMAP\n" + "="*30 + "\n\n"
        
        for phase in self.current_review.implementation_roadmap:
            roadmap += f"📅 {phase['phase']}\n"
            roadmap += "-" * (len(phase['phase']) + 3) + "\n"
            
            if phase['actions']:
                for action in phase['actions']:
                    roadmap += f"  • {action['action']}\n"
                    roadmap += f"    (Recommended by: {action['agent']})\n"
            else:
                roadmap += "  No specific actions identified for this phase.\n"
            
            roadmap += "\n"
        
        self.roadmap_text.delete("1.0", tk.END)
        self.roadmap_text.insert("1.0", roadmap)
        
        # Populate individual report combo
        session = self.review_manager.active_sessions.get(self.current_session)
        if session:
            agent_names = {
                "central_planning": "Central Planning Analyst",
                "labor_value": "Labor Value Theorist",
                "material_conditions": "Material Conditions Expert",
                "distribution": "Socialist Distribution Specialist", 
                "implementation": "Implementation Reviewer",
                "democracy": "Workers' Democracy Expert",
                "social_development": "Social Development Analyst"
            }
            
            available_reports = [agent_names.get(agent_id, agent_id) 
                               for agent_id in session.agent_reports.keys()]
            self.agent_report_combo['values'] = available_reports
            
            if available_reports:
                self.agent_report_combo.set(available_reports[0])
                self.show_individual_report()
    
    def show_individual_report(self, event=None):
        """Show individual agent report."""
        selected_agent_name = self.agent_report_var.get()
        if not selected_agent_name or not self.current_session:
            return
        
        # Find agent ID from name
        agent_names = {
            "Central Planning Analyst": "central_planning",
            "Labor Value Theorist": "labor_value",
            "Material Conditions Expert": "material_conditions",
            "Socialist Distribution Specialist": "distribution",
            "Implementation Reviewer": "implementation", 
            "Workers' Democracy Expert": "democracy",
            "Social Development Analyst": "social_development"
        }
        
        agent_id = agent_names.get(selected_agent_name)
        if not agent_id:
            return
        
        session = self.review_manager.active_sessions.get(self.current_session)
        if not session or agent_id not in session.agent_reports:
            return
        
        report = session.agent_reports[agent_id]
        
        individual_report = f"""INDIVIDUAL AGENT REPORT
{'='*40}

Agent: {report.agent_name}
Specialization: {report.agent_id.replace('_', ' ').title()}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}
Confidence Level: {report.confidence_level:.2f}/1.0

EXECUTIVE SUMMARY:
{report.executive_summary}

DETAILED ANALYSIS:
{report.detailed_analysis}

RISK ASSESSMENT:
{report.risk_assessment}

RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(report.recommendations, 1):
            individual_report += f"{i}. {rec}\n"
        
        if report.supporting_evidence:
            individual_report += f"\nSUPPORTING EVIDENCE:\n"
            for i, evidence in enumerate(report.supporting_evidence, 1):
                individual_report += f"{i}. {evidence}\n"
        
        self.individual_text.delete("1.0", tk.END)
        self.individual_text.insert("1.0", individual_report)
    
    def export_results(self, format_type):
        """Export results in specified format."""
        if not self.current_review:
            messagebox.showwarning("Warning", "No review results to export")
            return
        
        try:
            if format_type == "text":
                content = self.review_manager.export_review(self.current_session, "text")
                default_ext = ".txt"
                file_types = [("Text files", "*.txt"), ("All files", "*.*")]
            elif format_type == "json":
                content = self.review_manager.export_review(self.current_session, "json")
                default_ext = ".json"
                file_types = [("JSON files", "*.json"), ("All files", "*.*")]
            elif format_type == "summary":
                content = self.summary_text.get("1.0", tk.END)
                default_ext = ".txt"
                file_types = [("Text files", "*.txt"), ("All files", "*.*")]
            
            # Show export preview
            self.export_preview_text.delete("1.0", tk.END)
            self.export_preview_text.insert("1.0", content[:2000] + "..." if len(content) > 2000 else content)
            
            # Save file
            file_path = filedialog.asksaveasfilename(
                title=f"Export Review as {format_type.upper()}",
                defaultextension=default_ext,
                filetypes=file_types,
                initialname=f"economic_plan_review_{self.current_session}{default_ext}"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.export_status_label.config(text=f"✅ Exported to {os.path.basename(file_path)}", 
                                               foreground="green")
                self.status_bar.config(text=f"Review exported to {file_path}")
                messagebox.showinfo("Export Complete", f"Review exported successfully to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export review: {str(e)}")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = EconomicPlanReviewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()