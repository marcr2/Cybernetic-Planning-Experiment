"""
Inter-Agent Communication System

Facilitates structured communication between economic review agents,
enabling knowledge sharing, consensus building, and conflict resolution.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from queue import Queue, Empty
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""

    FINDING = "finding"
    QUESTION = "question"
    RESPONSE = "response"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    CONFLICT_ALERT = "conflict_alert"
    SYNTHESIS_REQUEST = "synthesis_request"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """Structured message between agents."""

    message_id: str
    sender_id: str
    recipient_id: str  # Can be "all" for broadcast
    message_type: MessageType
    priority: MessagePriority
    subject: str
    content: Dict[str, Any]
    timestamp: float
    requires_response: bool = False
    response_deadline: Optional[float] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationThread:
    """Represents a conversation thread between agents."""

    thread_id: str
    participants: Set[str]
    subject: str
    messages: List[AgentMessage]
    created_time: float
    last_activity: float
    status: str  # 'active', 'resolved', 'archived'


@dataclass
class ConsensusItem:
    """Represents an item being evaluated for consensus."""

    item_id: str
    topic: str
    statement: str
    agent_positions: Dict[str, Any]  # agent_id -> position/vote
    consensus_threshold: float
    created_time: float
    deadline: Optional[float]
    status: str  # 'pending', 'achieved', 'failed'


class CommunicationHub:
    """
    Central communication hub for inter-agent messaging.

    Manages message routing, conversation threads, consensus building,
    and conflict resolution between economic review agents.
    """

    def __init__(self):
        """Initialize the communication hub."""
        self.agents: Dict[str, Any] = {}  # agent_id -> agent reference
        self.message_queues: Dict[str, Queue] = defaultdict(Queue)
        self.conversations: Dict[str, ConversationThread] = {}
        self.consensus_items: Dict[str, ConsensusItem] = {}
        self.message_history: List[AgentMessage] = []

        # Statistics and monitoring
        self.message_stats = defaultdict(int)
        self.active_conversations = 0
        self.resolved_conflicts = 0

        # Thread safety
        self._lock = threading.RLock()

        # Auto-cleanup settings
        self.max_message_history = 1000
        self.conversation_timeout = 3600  # 1 hour

    def register_agent(self, agent_id: str, agent_reference: Any):
        """Register an agent with the communication hub."""
        with self._lock:
            self.agents[agent_id] = agent_reference
            logger.info(f"Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the communication hub."""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                # Clean up message queue
                if agent_id in self.message_queues:
                    del self.message_queues[agent_id]
                logger.info(f"Unregistered agent: {agent_id}")

    def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message between agents.

        Args:
            message: The message to send

        Returns:
            True if message was successfully queued
        """
        try:
            with self._lock:
                # Validate sender and recipient
                if message.sender_id not in self.agents:
                    logger.error(f"Unknown sender: {message.sender_id}")
                    return False

                if message.recipient_id != "all" and message.recipient_id not in self.agents:
                    logger.error(f"Unknown recipient: {message.recipient_id}")
                    return False

                # Add to message history
                self.message_history.append(message)
                self._cleanup_message_history()

                # Route message
                if message.recipient_id == "all":
                    # Broadcast to all agents except sender
                    for agent_id in self.agents:
                        if agent_id != message.sender_id:
                            self.message_queues[agent_id].put(message)
                else:
                    # Send to specific recipient
                    self.message_queues[message.recipient_id].put(message)

                # Update statistics
                self.message_stats[message.message_type.value] += 1
                self.message_stats["total"] += 1

                # Handle conversation threading
                self._handle_conversation_threading(message)

                logger.info(
                    f"Message sent: {message.sender_id} -> {message.recipient_id} ({message.message_type.value})"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False

    def receive_message(self, agent_id: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """
        Receive a message for a specific agent.

        Args:
            agent_id: ID of the receiving agent
            timeout: Maximum time to wait for a message

        Returns:
            Next message in queue or None if timeout
        """
        try:
            if agent_id not in self.agents:
                logger.error(f"Unknown agent: {agent_id}")
                return None

            queue = self.message_queues[agent_id]
            return queue.get(timeout=timeout)

        except Empty:
            return None
        except Exception as e:
            logger.error(f"Failed to receive message for {agent_id}: {str(e)}")
            return None

    def get_pending_messages(self, agent_id: str) -> List[AgentMessage]:
        """Get all pending messages for an agent."""
        messages = []
        try:
            queue = self.message_queues[agent_id]
            while not queue.empty():
                try:
                    message = queue.get_nowait()
                    messages.append(message)
                except Empty:
                    break
        except Exception as e:
            logger.error(f"Failed to get pending messages for {agent_id}: {str(e)}")

        return messages

    def start_conversation(self, initiator_id: str, participants: List[str], subject: str) -> str:
        """
        Start a new conversation thread.

        Args:
            initiator_id: ID of the agent starting the conversation
            participants: List of participant agent IDs
            subject: Conversation subject

        Returns:
            Conversation thread ID
        """
        thread_id = f"conv_{int(time.time())}_{initiator_id}"

        with self._lock:
            thread = ConversationThread(
                thread_id=thread_id,
                participants=set(participants + [initiator_id]),
                subject=subject,
                messages=[],
                created_time=time.time(),
                last_activity=time.time(),
                status="active",
            )

            self.conversations[thread_id] = thread
            self.active_conversations += 1

        logger.info(f"Started conversation: {thread_id} with {len(thread.participants)} participants")
        return thread_id

    def add_to_conversation(self, message: AgentMessage, thread_id: str):
        """Add a message to a conversation thread."""
        with self._lock:
            if thread_id in self.conversations:
                thread = self.conversations[thread_id]
                thread.messages.append(message)
                thread.last_activity = time.time()
                message.conversation_id = thread_id

    def request_consensus(
        self,
        requester_id: str,
        topic: str,
        statement: str,
        participants: List[str],
        threshold: float = 0.7,
        deadline_seconds: float = 3600,
    ) -> str:
        """
        Request consensus from multiple agents on a specific topic.

        Args:
            requester_id: ID of the agent requesting consensus
            topic: Topic category
            statement: Statement to achieve consensus on
            participants: List of agent IDs to participate
            threshold: Consensus threshold (0.0 to 1.0)
            deadline_seconds: Deadline in seconds from now

        Returns:
            Consensus item ID
        """
        item_id = f"consensus_{int(time.time())}_{requester_id}"
        deadline = time.time() + deadline_seconds if deadline_seconds else None

        consensus_item = ConsensusItem(
            item_id=item_id,
            topic=topic,
            statement=statement,
            agent_positions={},
            consensus_threshold=threshold,
            created_time=time.time(),
            deadline=deadline,
            status="pending",
        )

        with self._lock:
            self.consensus_items[item_id] = consensus_item

        # Send consensus request messages
        for participant_id in participants:
            if participant_id != requester_id and participant_id in self.agents:
                message = AgentMessage(
                    message_id=f"consensus_req_{item_id}_{participant_id}",
                    sender_id=requester_id,
                    recipient_id=participant_id,
                    message_type=MessageType.CONSENSUS_REQUEST,
                    priority=MessagePriority.HIGH,
                    subject=f"Consensus Request: {topic}",
                    content={
                        "consensus_id": item_id,
                        "topic": topic,
                        "statement": statement,
                        "threshold": threshold,
                        "deadline": deadline,
                    },
                    timestamp=time.time(),
                    requires_response=True,
                    response_deadline=deadline,
                )
                self.send_message(message)

        logger.info(f"Requested consensus: {item_id} from {len(participants)} agents")
        return item_id

    def submit_consensus_position(self, agent_id: str, consensus_id: str, position: Dict[str, Any]) -> bool:
        """
        Submit an agent's position on a consensus item.

        Args:
            agent_id: ID of the responding agent
            consensus_id: ID of the consensus item
            position: Agent's position (should include 'agreement' and 'reasoning')

        Returns:
            True if position was recorded successfully
        """
        with self._lock:
            if consensus_id not in self.consensus_items:
                logger.error(f"Unknown consensus item: {consensus_id}")
                return False

            consensus_item = self.consensus_items[consensus_id]
            if consensus_item.status != "pending":
                logger.warning(f"Consensus item {consensus_id} is not pending")
                return False

            # Record position
            consensus_item.agent_positions[agent_id] = position

            # Check if consensus is achieved
            self._evaluate_consensus(consensus_id)

        logger.info(f"Recorded consensus position from {agent_id} for {consensus_id}")
        return True

    def detect_conflicts(self, agent_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent findings.

        Args:
            agent_reports: Dictionary of agent reports

        Returns:
            List of detected conflicts
        """
        conflicts = []

        try:
            # Compare confidence levels for similar topics
            confidence_conflicts = self._detect_confidence_conflicts(agent_reports)
            conflicts.extend(confidence_conflicts)

            # Compare recommendations for contradictions
            recommendation_conflicts = self._detect_recommendation_conflicts(agent_reports)
            conflicts.extend(recommendation_conflicts)

            # Compare risk assessments for inconsistencies
            risk_conflicts = self._detect_risk_conflicts(agent_reports)
            conflicts.extend(risk_conflicts)

            logger.info(f"Detected {len(conflicts)} potential conflicts")

        except Exception as e:
            logger.error(f"Failed to detect conflicts: {str(e)}")

        return conflicts

    def resolve_conflict(self, conflict: Dict[str, Any], resolution_strategy: str = "consensus") -> Dict[str, Any]:
        """
        Attempt to resolve a conflict between agents.

        Args:
            conflict: Conflict description
            resolution_strategy: Strategy to use ('consensus', 'expert', 'majority')

        Returns:
            Resolution result
        """
        try:
            if resolution_strategy == "consensus":
                return self._resolve_by_consensus(conflict)
            elif resolution_strategy == "expert":
                return self._resolve_by_expert(conflict)
            elif resolution_strategy == "majority":
                return self._resolve_by_majority(conflict)
            else:
                return {"status": "failed", "reason": f"Unknown strategy: {resolution_strategy}"}

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {str(e)}")
            return {"status": "failed", "reason": str(e)}

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        with self._lock:
            return {
                "registered_agents": len(self.agents),
                "total_messages": self.message_stats["total"],
                "messages_by_type": dict(self.message_stats),
                "active_conversations": self.active_conversations,
                "total_conversations": len(self.conversations),
                "pending_consensus_items": len([c for c in self.consensus_items.values() if c.status == "pending"]),
                "resolved_conflicts": self.resolved_conflicts,
            }

    def _handle_conversation_threading(self, message: AgentMessage):
        """Handle conversation threading for messages."""
        if message.conversation_id:
            self.add_to_conversation(message, message.conversation_id)

    def _cleanup_message_history(self):
        """Clean up old messages from history."""
        if len(self.message_history) > self.max_message_history:
            # Keep only the most recent messages
            self.message_history = self.message_history[-self.max_message_history :]

    def _evaluate_consensus(self, consensus_id: str):
        """Evaluate whether consensus has been achieved."""
        consensus_item = self.consensus_items[consensus_id]

        if not consensus_item.agent_positions:
            return

        # Calculate agreement percentage
        agreements = [pos.get("agreement", 0) for pos in consensus_item.agent_positions.values()]
        if agreements:
            avg_agreement = sum(agreements) / len(agreements)

            if avg_agreement >= consensus_item.consensus_threshold:
                consensus_item.status = "achieved"
                logger.info(f"Consensus achieved for {consensus_id}: {avg_agreement:.2f}")
            elif consensus_item.deadline and time.time() > consensus_item.deadline:
                consensus_item.status = "failed"
                logger.info(f"Consensus failed for {consensus_id}: deadline exceeded")

    def _detect_confidence_conflicts(self, agent_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts based on confidence levels."""
        conflicts = []

        # Find agents with very different confidence levels
        confidence_levels = {}
        for agent_id, report in agent_reports.items():
            if hasattr(report, "confidence_level"):
                confidence_levels[agent_id] = report.confidence_level

        if len(confidence_levels) >= 2:
            min_conf = min(confidence_levels.values())
            max_conf = max(confidence_levels.values())

            if max_conf - min_conf > 0.4:  # Significant confidence gap
                conflicts.append(
                    {
                        "type": "confidence_disparity",
                        "description": f"Large confidence gap: {min_conf:.2f} to {max_conf:.2f}",
                        "agents": list(confidence_levels.keys()),
                        "severity": "medium",
                    }
                )

        return conflicts

    def _detect_recommendation_conflicts(self, agent_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts in recommendations."""
        conflicts = []

        # Collect all recommendations
        all_recommendations = {}
        for agent_id, report in agent_reports.items():
            if hasattr(report, "recommendations"):
                all_recommendations[agent_id] = report.recommendations

        # Look for contradictory recommendations (simplified)
        contradiction_keywords = [
            ("increase", "decrease"),
            ("expand", "reduce"),
            ("accelerate", "slow"),
            ("prioritize", "deprioritize"),
        ]

        for keyword1, keyword2 in contradiction_keywords:
            agents_with_kw1 = []
            agents_with_kw2 = []

            for agent_id, recommendations in all_recommendations.items():
                for rec in recommendations:
                    if keyword1 in rec.lower():
                        agents_with_kw1.append(agent_id)
                    if keyword2 in rec.lower():
                        agents_with_kw2.append(agent_id)

            if agents_with_kw1 and agents_with_kw2:
                conflicts.append(
                    {
                        "type": "recommendation_contradiction",
                        "description": f"Contradictory recommendations: {keyword1} vs {keyword2}",
                        "agents": agents_with_kw1 + agents_with_kw2,
                        "severity": "high",
                    }
                )

        return conflicts

    def _detect_risk_conflicts(self, agent_reports: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts in risk assessments."""
        conflicts = []

        # Simplified risk conflict detection
        # In practice, this would use more sophisticated NLP

        risk_levels = {}
        for agent_id, report in agent_reports.items():
            if hasattr(report, "risk_assessment"):
                risk_text = report.risk_assessment.lower()
                if "high risk" in risk_text or "critical" in risk_text:
                    risk_levels[agent_id] = "high"
                elif "low risk" in risk_text or "minimal" in risk_text:
                    risk_levels[agent_id] = "low"
                else:
                    risk_levels[agent_id] = "medium"

        # Check for conflicting risk assessments
        if len(set(risk_levels.values())) > 1:
            high_risk_agents = [a for a, r in risk_levels.items() if r == "high"]
            low_risk_agents = [a for a, r in risk_levels.items() if r == "low"]

            if high_risk_agents and low_risk_agents:
                conflicts.append(
                    {
                        "type": "risk_assessment_conflict",
                        "description": "Conflicting risk assessments between agents",
                        "agents": high_risk_agents + low_risk_agents,
                        "severity": "medium",
                    }
                )

        return conflicts

    def _resolve_by_consensus(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by seeking consensus."""
        # Start a consensus process
        consensus_id = self.request_consensus(
            requester_id="system",
            topic=conflict["type"],
            statement=conflict["description"],
            participants=conflict["agents"],
            threshold=0.6,
        )

        return {"status": "in_progress", "method": "consensus", "consensus_id": consensus_id}

    def _resolve_by_expert(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by deferring to the most expert agent."""
        # Simplified: choose agent with highest confidence
        # In practice, would consider domain expertise
        return {"status": "resolved", "method": "expert_decision", "resolution": "Deferred to most confident agent"}

    def _resolve_by_majority(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by majority vote."""
        return {"status": "resolved", "method": "majority_vote", "resolution": "Majority position adopted"}


# Utility functions for message creation
def create_finding_message(sender_id: str, recipient_id: str, finding: Dict[str, Any]) -> AgentMessage:
    """Create a finding sharing message."""
    return AgentMessage(
        message_id=f"finding_{int(time.time())}_{sender_id}",
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MessageType.FINDING,
        priority=MessagePriority.NORMAL,
        subject=f"Finding: {finding.get('topic', 'Economic Analysis')}",
        content=finding,
        timestamp=time.time(),
    )


def create_question_message(
    sender_id: str, recipient_id: str, question: str, context: Dict[str, Any] = None
) -> AgentMessage:
    """Create a question message."""
    return AgentMessage(
        message_id=f"question_{int(time.time())}_{sender_id}",
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=MessageType.QUESTION,
        priority=MessagePriority.NORMAL,
        subject=f"Question from {sender_id}",
        content={"question": question, "context": context or {}},
        timestamp=time.time(),
        requires_response=True,
    )


def create_coordination_message(
    sender_id: str, recipients: List[str], coordination_request: Dict[str, Any]
) -> List[AgentMessage]:
    """Create coordination messages for multiple recipients."""
    messages = []

    for recipient_id in recipients:
        message = AgentMessage(
            message_id=f"coord_{int(time.time())}_{sender_id}_{recipient_id}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.COORDINATION,
            priority=MessagePriority.HIGH,
            subject=f"Coordination Request: {coordination_request.get('task', 'Unknown')}",
            content=coordination_request,
            timestamp=time.time(),
            requires_response=True,
        )
        messages.append(message)

    return messages
