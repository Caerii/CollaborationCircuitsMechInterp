"""
Response Parser for Reasoning Models

Extracts structured information from model responses that use <think> tags.
Qwen3-4B and similar reasoning models produce responses like:

    <think>
    Let me reason through this step by step...
    Alice put the ball in the drawer, then left.
    Bob moved it to the basket.
    Alice didn't see this, so she still thinks it's in the drawer.
    </think>
    
    drawer

This module provides utilities to:
1. Extract the final answer (text after </think>)
2. Extract the reasoning process (text inside <think>)
3. Validate response format
4. Handle edge cases (no tags, partial tags, etc.)
"""

import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Structured representation of a parsed model response."""
    raw_response: str
    reasoning: Optional[str]  # Content inside <think> tags
    answer: Optional[str]  # Content after </think>
    has_think_tags: bool
    is_valid: bool
    confidence: float  # Estimate based on response quality
    
    def to_dict(self) -> Dict:
        return {
            "raw_response": self.raw_response,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "has_think_tags": self.has_think_tags,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
        }


class ResponseParser:
    """
    Parser for reasoning model responses with <think> tags.
    
    Example:
        parser = ResponseParser()
        
        response = "<think>Alice doesn't know...</think>drawer"
        parsed = parser.parse(response)
        
        print(parsed.answer)  # "drawer"
        print(parsed.reasoning)  # "Alice doesn't know..."
    """
    
    # Patterns for extracting think content
    THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    
    # Pattern for finding answer after think tags
    AFTER_THINK_PATTERN = re.compile(r'</think>\s*(.+)', re.DOTALL | re.IGNORECASE)
    
    # Common answer words to look for
    LOCATION_WORDS = {
        "drawer", "basket", "box", "container", "cupboard", "shelf",
        "bag", "pocket", "table", "desk", "cabinet", "closet"
    }
    
    BOOLEAN_WORDS = {"yes", "no", "true", "false"}
    
    AGENT_WORDS = {"alice", "bob", "carol", "dave", "eve", "user", "assistant"}
    
    def __init__(self, expected_answers: Optional[List[str]] = None):
        """
        Initialize parser.
        
        Args:
            expected_answers: Optional list of expected answer tokens for validation
        """
        self.expected_answers = set(a.lower() for a in expected_answers) if expected_answers else None
    
    def parse(self, response: str) -> ParsedResponse:
        """
        Parse a model response, extracting reasoning and answer.
        
        Args:
            response: Raw model response string
            
        Returns:
            ParsedResponse with extracted components
        """
        if not response or not response.strip():
            return ParsedResponse(
                raw_response=response,
                reasoning=None,
                answer=None,
                has_think_tags=False,
                is_valid=False,
                confidence=0.0
            )
        
        response = response.strip()
        
        # Check for think tags
        think_match = self.THINK_PATTERN.search(response)
        has_think_tags = think_match is not None
        
        reasoning = None
        answer = None
        
        if has_think_tags:
            # Extract reasoning
            reasoning = think_match.group(1).strip()
            
            # Extract answer after </think>
            after_match = self.AFTER_THINK_PATTERN.search(response)
            if after_match:
                answer = self._clean_answer(after_match.group(1))
            else:
                # Try to find answer at end of response
                answer = self._extract_answer_from_end(response)
        else:
            # No think tags - entire response might be the answer
            answer = self._extract_answer_from_end(response)
        
        # Validate
        is_valid = answer is not None and len(answer) > 0
        
        # Estimate confidence
        confidence = self._estimate_confidence(reasoning, answer, has_think_tags)
        
        return ParsedResponse(
            raw_response=response,
            reasoning=reasoning,
            answer=answer,
            has_think_tags=has_think_tags,
            is_valid=is_valid,
            confidence=confidence
        )
    
    def _clean_answer(self, text: str) -> str:
        """Clean and extract the core answer from text."""
        if not text:
            return ""
        
        # Remove common prefixes
        text = text.strip()
        prefixes = ["answer:", "the answer is", "answer is", "my answer:", "final answer:"]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Get first word/line as answer
        lines = text.strip().split('\n')
        first_line = lines[0].strip()
        
        # Get first meaningful word
        words = first_line.split()
        if words:
            # Clean punctuation
            answer = words[0].strip('.,!?:;"\'-')
            return answer
        
        return first_line
    
    def _extract_answer_from_end(self, response: str) -> Optional[str]:
        """Try to extract an answer from the end of a response without think tags."""
        lines = response.strip().split('\n')
        
        # Try last few lines
        for line in reversed(lines[-3:]):
            line = line.strip()
            if not line:
                continue
            
            # Check if it looks like an answer
            first_word = line.split()[0].lower().strip('.,!?') if line.split() else ""
            
            # Check against known answer types
            all_known = self.LOCATION_WORDS | self.BOOLEAN_WORDS | self.AGENT_WORDS
            if self.expected_answers:
                all_known = all_known | self.expected_answers
            
            if first_word in all_known:
                return first_word
            
            # If line is short, treat as answer
            if len(line) < 30 and len(line.split()) <= 3:
                return self._clean_answer(line)
        
        return None
    
    def _estimate_confidence(
        self,
        reasoning: Optional[str],
        answer: Optional[str],
        has_think_tags: bool
    ) -> float:
        """Estimate confidence in the parsed response."""
        confidence = 0.0
        
        # Having think tags is good
        if has_think_tags:
            confidence += 0.3
        
        # Having reasoning content is good
        if reasoning and len(reasoning) > 50:
            confidence += 0.3
        elif reasoning and len(reasoning) > 10:
            confidence += 0.2
        
        # Having a clear answer is good
        if answer:
            confidence += 0.2
            
            # Known answer words boost confidence
            answer_lower = answer.lower()
            all_known = self.LOCATION_WORDS | self.BOOLEAN_WORDS | self.AGENT_WORDS
            if answer_lower in all_known:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def extract_answer_token(self, response: str, options: List[str]) -> Tuple[Optional[str], float]:
        """
        Extract which option the model selected.
        
        Args:
            response: Model response
            options: List of valid options (e.g., ["drawer", "basket"])
            
        Returns:
            Tuple of (selected option, confidence) or (None, 0.0)
        """
        parsed = self.parse(response)
        
        if not parsed.answer:
            return None, 0.0
        
        answer_lower = parsed.answer.lower()
        
        # Exact match
        for opt in options:
            if opt.lower() == answer_lower:
                return opt, parsed.confidence
        
        # Partial match (answer contains option)
        for opt in options:
            if opt.lower() in answer_lower or answer_lower in opt.lower():
                return opt, parsed.confidence * 0.8
        
        # Check full response for options
        response_lower = response.lower()
        matches = []
        for opt in options:
            if opt.lower() in response_lower:
                # Find last occurrence (usually the final answer)
                pos = response_lower.rfind(opt.lower())
                matches.append((opt, pos))
        
        if matches:
            # Return option with latest position
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0], 0.5
        
        return None, 0.0
    
    def batch_parse(self, responses: List[str]) -> List[ParsedResponse]:
        """Parse multiple responses."""
        return [self.parse(r) for r in responses]


def extract_final_answer(response: str, options: Optional[List[str]] = None) -> str:
    """
    Convenience function to extract final answer from response.
    
    Args:
        response: Model response with potential <think> tags
        options: Optional list of valid options
        
    Returns:
        Extracted answer string
    """
    parser = ResponseParser(expected_answers=options)
    
    if options:
        answer, _ = parser.extract_answer_token(response, options)
        return answer or ""
    else:
        parsed = parser.parse(response)
        return parsed.answer or ""

