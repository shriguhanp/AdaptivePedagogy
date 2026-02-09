
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.logging import get_logger

logger = get_logger("FlashcardManager")

# Models for internal use and persistence
class FlashcardItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    front: str
    back: str
    is_starred: bool = False
    review_count: int = 0
    last_reviewed: Optional[datetime] = None

class FlashcardSet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"  # Placeholder for single-user mode
    kb_name: str
    topic: str
    created_at: datetime = Field(default_factory=datetime.now)
    cards: List[FlashcardItem]

class FlashcardManager:
    """
    Manages persistence of flashcard sets using JSON files.
    
    Directory Structure:
    data/user/flashcards/
      ├── {set_id}.json
      ├── {set_id}.json
      ...
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Default to project_root/data/user/flashcards
            # Assuming project root is 3 levels up from here similar to knowledge manager
            project_root = Path(__file__).parent.parent.parent.parent
            self.base_dir = project_root / "data" / "user" / "flashcards"
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_file_path(self, set_id: str) -> Path:
        return self.base_dir / f"{set_id}.json"

    def create_set(self, kb_name: str, topic: str, cards_data: List[Dict[str, str]]) -> FlashcardSet:
        """Create and save a new flashcard set."""
        cards = [
            FlashcardItem(front=c["front"], back=c["back"]) 
            for c in cards_data
        ]
        
        flashcard_set = FlashcardSet(
            kb_name=kb_name,
            topic=topic,
            cards=cards
        )
        
        self.save_set(flashcard_set)
        logger.info(f"Created flashcard set {flashcard_set.id} with {len(cards)} cards")
        return flashcard_set

    def save_set(self, flashcard_set: FlashcardSet):
        """Persist a set to disk."""
        file_path = self._get_file_path(flashcard_set.id)
        with open(file_path, "w") as f:
            f.write(flashcard_set.model_dump_json(indent=2))

    def get_set(self, set_id: str) -> Optional[FlashcardSet]:
        """Retrieve a set by ID."""
        file_path = self._get_file_path(set_id)
        if not file_path.exists():
            return None
            
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return FlashcardSet(**data)
        except Exception as e:
            logger.error(f"Failed to load flashcard set {set_id}: {e}")
            return None

    def list_sets(self, user_id: str = "default_user") -> List[FlashcardSet]:
        """List all sets (optionally filtered by user, though currently all files are assumed compliant)."""
        sets = []
        for file_path in self.base_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                fs = FlashcardSet(**data)
                # Filter by user_id if we have multi-user logic later, for now just matching default
                if fs.user_id == user_id:
                    sets.append(fs)
            except Exception as e:
                logger.warning(f"Error reading flashcard file {file_path}: {e}")
                
        # Sort by created_at desc
        sets.sort(key=lambda x: x.created_at, reverse=True)
        return sets

    def delete_set(self, set_id: str) -> bool:
        """Delete a set."""
        file_path = self._get_file_path(set_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def update_card_review(self, set_id: str, card_id: str) -> Optional[FlashcardSet]:
        """Mark a card as reviewed."""
        flashcard_set = self.get_set(set_id)
        if not flashcard_set:
            return None
            
        for card in flashcard_set.cards:
            if card.id == card_id:
                card.review_count += 1
                card.last_reviewed = datetime.now()
                self.save_set(flashcard_set)
                return flashcard_set
                
        return None

    def toggle_card_star(self, set_id: str, card_id: str) -> Optional[FlashcardSet]:
        """Toggle star status of a card."""
        flashcard_set = self.get_set(set_id)
        if not flashcard_set:
            return None
            
        for card in flashcard_set.cards:
            if card.id == card_id:
                card.is_starred = not card.is_starred
                self.save_set(flashcard_set)
                return flashcard_set
                
        return None
