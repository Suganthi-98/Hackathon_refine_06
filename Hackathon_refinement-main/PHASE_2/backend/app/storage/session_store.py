"""
Session Store

In-memory storage for project sessions.
For hackathon: one project per session.
For production: replace with Redis + session tokens.
"""

from typing import Dict, Optional
from datetime import datetime
from threading import Lock

from app.domain.models import ProjectState


class Session:
    """Single project session."""
    
    def __init__(self, session_id: str, project_state: ProjectState):
        self.session_id = session_id
        self.project_state = project_state
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.descoped_item_ids = set()  # For scope change tracking
        # Lazily populated by SessionStore.get_analysis() — holds the single
        # computed truth (ProjectAnalysis) so every route reads the same numbers.
        self._analysis = None
    
    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.utcnow()

    def invalidate_analysis(self) -> None:
        """
        Drop the cached ProjectAnalysis so it is recomputed on the next
        get_analysis() call.  Must be called whenever project_state is mutated
        (e.g. scope change, descope).
        """
        self._analysis = None


class SessionStore:
    """Thread-safe in-memory session storage."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize store."""
        if self._initialized:
            return
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
        self._initialized = True
    
    def create_session(self, project_state: ProjectState) -> str:
        """
        Create a new session for a project.
        
        Args:
            project_state: ProjectState to store
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = project_state.project_id
        session = Session(session_id, project_state)
        
        with self._lock:
            self._sessions[session_id] = session
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def get_project_state(self, session_id: str) -> Optional[ProjectState]:
        """
        Retrieve project state from session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ProjectState or None if not found
        """
        session = self.get_session(session_id)
        return session.project_state if session else None

    def get_analysis(self, session_id: str, simulation_count: int = 1000):
        """
        Return the cached ProjectAnalysis for this session, building it on
        the first call.

        This is the single point of truth for all engine outputs.  Every API
        route should call this instead of constructing engines independently.

        Args:
            session_id:        Session identifier.
            simulation_count:  Monte Carlo iterations (default 1000).

        Returns:
            ProjectAnalysis or None if the session does not exist.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        if session._analysis is None:
            # Import here to avoid a module-level circular dependency.
            from app.engines.project_analysis import ProjectAnalysis
            session._analysis = ProjectAnalysis.build(
                session.project_state,
                simulation_count=simulation_count,
            )

        return session._analysis

    def invalidate_analysis(self, session_id: str) -> None:
        """
        Drop the cached ProjectAnalysis for a session so it is rebuilt on
        the next get_analysis() call.

        Call this whenever project_state is mutated (scope change, descope,
        blocker resolution, etc.) so routes don't serve stale numbers.

        Args:
            session_id: Session identifier.
        """
        session = self.get_session(session_id)
        if session is not None:
            session.invalidate_analysis()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def list_sessions(self) -> list:
        """
        List all active sessions.
        
        Returns:
            List of (session_id, project_name) tuples
        """
        with self._lock:
            return [
                (sid, s.project_state.project_info.project_name)
                for sid, s in self._sessions.items()
            ]
    
    def clear_all(self) -> None:
        """Clear all sessions (for testing)."""
        with self._lock:
            self._sessions.clear()


# Global singleton instance
store = SessionStore()
