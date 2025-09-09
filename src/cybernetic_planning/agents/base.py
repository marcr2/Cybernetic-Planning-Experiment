"""
Base Agent Class

Provides the foundation for all specialized agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""

    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.

    Provides common functionality for agent communication, state management,
    and task execution.
    """

    def __init__(self, agent_id: str, name: str):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        self.agent_id = agent_id
        self.name = name
        self.state = {}
        self.message_queue = []
        self.active = True

    def send_message(self, recipient: str, message_type: str, content: Dict[str, Any]) -> None:
        """
        Send a message to another agent.

        Args:
            recipient: ID of the recipient agent
            message_type: Type of message
            content: Message content
        """
        import time

        message = AgentMessage(
            sender=self.agent_id, recipient=recipient, message_type=message_type, content=content, timestamp=time.time()
        )
        # In a real implementation, this would be sent through a message broker
        # For now, we'll just store it locally
        self.message_queue.append(message)

    def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive the next message from the queue.

        Returns:
            Next message in queue, or None if empty
        """
        if self.message_queue:
            return self.message_queue.pop(0)
        return None

    def update_state(self, key: str, value: Any) -> None:
        """
        Update agent state.

        Args:
            key: State key
            value: State value
        """
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get agent state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return self.state.get(key, default)

    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.

        Args:
            task: Task description and parameters

        Returns:
            Task results
        """

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.

        Returns:
            List of capability strings
        """

    def activate(self) -> None:
        """Activate the agent."""
        self.active = True

    def deactivate(self) -> None:
        """Deactivate the agent."""
        self.active = False

    def is_active(self) -> bool:
        """Check if agent is active."""
        return self.active

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.

        Returns:
            Dictionary with agent status
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "active": self.active,
            "state_keys": list(self.state.keys()),
            "message_queue_length": len(self.message_queue),
            "capabilities": self.get_capabilities(),
        }
