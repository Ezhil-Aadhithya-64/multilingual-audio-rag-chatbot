"""
Guardrails Engine - Response validation and safety checks
Ensures responses meet quality, safety, and grounding requirements
"""

import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class GuardrailsEngine:
    """
    Validates responses before delivery to users.
    Checks confidence, grounding, hallucination, and safety.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        grounding_threshold: float = 0.7,
        enable_pii_detection: bool = True
    ):
        """
        Initialize guardrails engine.

        Args:
            confidence_threshold: Minimum confidence for responses
            grounding_threshold: Minimum context overlap required
            enable_pii_detection: Enable PII detection and filtering
        """
        self.confidence_threshold = confidence_threshold
        self.grounding_threshold = grounding_threshold
        self.enable_pii_detection = enable_pii_detection

        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }

    def validate(
        self,
        response: str,
        confidence: float,
        retrieved_context: List[str],
        query: str,
        retrieval_scores: Optional[List[float]] = None,
        reranker_scores: Optional[List[float]] = None,
        retry_count: int = 0
    ) -> Tuple[bool, str, Dict]:
        """
        Validate response against all guardrails.

        Args:
            response: Generated response text
            confidence: Response confidence score
            retrieved_context: Context used for generation
            query: Original user query
            retrieval_scores: Optional similarity scores from retrieval
            reranker_scores: Optional scores from reranker
            retry_count: Current retry attempt number

        Returns:
            Tuple of (is_valid, final_response, validation_metadata)
        """
        logger.info(f"Running guardrails validation (retry: {retry_count})...")

        validation_results = {
            "confidence_check": False,
            "grounding_check": False,
            "hallucination_check": False,
            "safety_check": False,
            "pii_check": False,
            "context_sufficiency_check": False,
            "retry_count": retry_count
        }

        # 1. Context availability check (FIRST)
        if not retrieved_context or len(retrieved_context) == 0:
            logger.warning("No context available")
            return False, self._generate_no_context_response(), {
                **validation_results,
                "failure_reason": "no_context",
                "retry_recommended": False,  # Can't retry without context
                "feedback": "No relevant context was retrieved from the knowledge base"
            }
        validation_results["context_sufficiency_check"] = True
        
        # 2. Context quality check (NEW)
        context_quality = self._assess_context_quality(
            retrieved_context, 
            retrieval_scores, 
            reranker_scores
        )
        
        if context_quality < 0.3:
            logger.warning(f"Low context quality: {context_quality:.2f}")
            return False, self._generate_insufficient_context_response(), {
                **validation_results,
                "failure_reason": "insufficient_context_quality",
                "context_quality": context_quality,
                "retry_recommended": False,  # Context quality won't improve on retry
                "feedback": f"Retrieved context quality is too low ({context_quality:.2f})"
            }

        # 3. Enhanced confidence check
        enhanced_confidence = self._calculate_enhanced_confidence(
            base_confidence=confidence,
            context_quality=context_quality,
            response_length=len(response),
            context_length=sum(len(c) for c in retrieved_context)
        )
        
        if enhanced_confidence < self.confidence_threshold:
            logger.warning(f"Low enhanced confidence: {enhanced_confidence:.2f}")
            return False, self._generate_low_confidence_response(), {
                **validation_results,
                "failure_reason": "low_confidence",
                "base_confidence": confidence,
                "enhanced_confidence": enhanced_confidence,
                "context_quality": context_quality,
                "retry_recommended": True,  # Retry may improve confidence
                "feedback": self._generate_confidence_feedback(
                    enhanced_confidence, 
                    context_quality,
                    response,
                    retrieved_context
                )
            }
        validation_results["confidence_check"] = True

        # 4. Grounding check
        grounding_score = self._calculate_grounding_score(response, retrieved_context)
        if grounding_score < self.grounding_threshold:
            logger.warning(f"Low grounding score: {grounding_score:.2f}")
            return False, self._generate_ungrounded_response(), {
                **validation_results,
                "failure_reason": "insufficient_grounding",
                "grounding_score": grounding_score,
                "retry_recommended": True,  # Retry with better prompt may improve grounding
                "feedback": self._generate_grounding_feedback(
                    grounding_score,
                    response,
                    retrieved_context
                )
            }
        validation_results["grounding_check"] = True

        # 5. Hallucination detection
        has_hallucination, hallucination_indicators = self._detect_hallucination(
            response, retrieved_context
        )
        if has_hallucination:
            logger.warning(f"Hallucination detected: {hallucination_indicators}")
            return False, self._generate_safe_fallback(retrieved_context), {
                **validation_results,
                "failure_reason": "hallucination_detected",
                "indicators": hallucination_indicators,
                "retry_recommended": True,  # Retry with stricter prompt
                "feedback": self._generate_hallucination_feedback(hallucination_indicators)
            }
        validation_results["hallucination_check"] = True

        # 6. Safety check
        is_safe, safety_issues = self._check_safety(response)
        if not is_safe:
            logger.warning(f"Safety issues detected: {safety_issues}")
            return False, "I cannot provide that information.", {
                **validation_results,
                "failure_reason": "safety_violation",
                "issues": safety_issues,
                "retry_recommended": False,  # Don't retry safety violations
                "feedback": "Response contains sensitive information that should not be shared"
            }
        validation_results["safety_check"] = True

        # 7. PII detection and filtering
        if self.enable_pii_detection:
            response, pii_found = self._filter_pii(response)
            validation_results["pii_check"] = True
            if pii_found:
                logger.info(f"PII filtered: {pii_found}")

        # All checks passed
        logger.info("All guardrails passed")
        return True, response, {
            **validation_results,
            "grounding_score": grounding_score,
            "base_confidence": confidence,
            "enhanced_confidence": enhanced_confidence,
            "context_quality": context_quality,
            "passed": True,
            "retry_recommended": False
        }

    def _calculate_grounding_score(
        self,
        response: str,
        context: List[str]
    ) -> float:
        """
        Calculate how well response is grounded in context.
        Uses token overlap as proxy for grounding.

        Args:
            response: Generated response
            context: Retrieved context documents

        Returns:
            Grounding score (0-1)
        """
        if not context:
            return 0.0

        # Tokenize response and context
        response_tokens = set(response.lower().split())
        context_tokens = set()
        for doc in context:
            context_tokens.update(doc.lower().split())

        # Calculate overlap
        if len(response_tokens) == 0:
            return 0.0

        overlap = len(response_tokens.intersection(context_tokens))
        score = overlap / len(response_tokens)

        return min(score, 1.0)

    def _detect_hallucination(
        self,
        response: str,
        context: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Detect potential hallucinations in response.

        Args:
            response: Generated response
            context: Retrieved context

        Returns:
            Tuple of (has_hallucination, indicators)
        """
        indicators = []

        # Check for common hallucination patterns
        hallucination_patterns = [
            (r"according to (my|general) knowledge", "unsupported_claim"),
            (r"it is (widely|commonly) known", "unsupported_claim"),
            (r"studies (show|suggest|indicate)", "unsupported_citation"),
            (r"research (shows|suggests|indicates)", "unsupported_citation"),
            (r"experts (say|believe|think)", "unsupported_authority"),
        ]

        for pattern, indicator_type in hallucination_patterns:
            if re.search(pattern, response.lower()):
                indicators.append(indicator_type)

        # Check for specific numbers/dates not in context
        response_numbers = re.findall(r'\b\d+\b', response)
        context_text = " ".join(context)
        for number in response_numbers:
            if number not in context_text and len(number) > 2:
                indicators.append(f"unsupported_number:{number}")

        return len(indicators) > 0, indicators

    def _check_safety(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check response for safety issues.

        Args:
            response: Generated response

        Returns:
            Tuple of (is_safe, issues)
        """
        issues = []
        response_lower = response.lower()

        # Check for harmful content patterns
        harmful_patterns = [
            "password is",
            "api key",
            "secret key",
            "private key",
            "access token"
        ]

        for pattern in harmful_patterns:
            if pattern in response_lower:
                issues.append(f"sensitive_info:{pattern}")

        return len(issues) == 0, issues

    def _filter_pii(self, text: str) -> Tuple[str, List[str]]:
        """
        Detect and filter PII from text.

        Args:
            text: Input text

        Returns:
            Tuple of (filtered_text, pii_types_found)
        """
        filtered_text = text
        pii_found = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_found.append(pii_type)
                # Replace with placeholder
                filtered_text = re.sub(
                    pattern,
                    f"[{pii_type.upper()}_REDACTED]",
                    filtered_text
                )

        return filtered_text, pii_found

    def _generate_low_confidence_response(self) -> str:
        """Generate response for low confidence scenarios."""
        return (
            "I don't have enough confidence to provide an accurate answer. "
            "Could you please rephrase your question or provide more context?"
        )

    def _generate_no_context_response(self) -> str:
        """Generate response when no context is available."""
        return (
            "I couldn't find relevant information in the knowledge base to answer "
            "your question. Please try rephrasing or ask about topics covered in "
            "the documentation."
        )

    def _generate_ungrounded_response(self) -> str:
        """Generate response for insufficient grounding."""
        return (
            "I cannot provide a well-grounded answer based on the available "
            "documentation. Please ask a more specific question or check if "
            "the information exists in the knowledge base."
        )

    def _generate_safe_fallback(self, context: List[str]) -> str:
        """Generate safe fallback response using only context."""
        if not context:
            return self._generate_no_context_response()

        # Return first context chunk as safe fallback
        return (
            "Based on the available documentation:\n\n"
            f"{context[0][:300]}...\n\n"
            "Please let me know if you need more specific information."
        )
    
    def _assess_context_quality(
        self,
        context: List[str],
        retrieval_scores: Optional[List[float]] = None,
        reranker_scores: Optional[List[float]] = None
    ) -> float:
        """
        Assess quality of retrieved context.
        
        Args:
            context: Retrieved context documents
            retrieval_scores: Similarity scores from retrieval (0-1 range)
            reranker_scores: Scores from reranker (can be negative, need normalization)
            
        Returns:
            Context quality score (0-1)
        """
        if not context:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Number of documents (more is better, up to a point)
        doc_count_score = min(len(context) / 5.0, 1.0)
        quality_factors.append(doc_count_score)
        
        # Factor 2: Average document length (longer is better, indicates substance)
        avg_length = sum(len(doc) for doc in context) / len(context)
        length_score = min(avg_length / 500.0, 1.0)  # 500 chars is good
        quality_factors.append(length_score)
        
        # Factor 3: Retrieval scores (if available, already in 0-1 range)
        if retrieval_scores:
            avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
            quality_factors.append(avg_retrieval_score)
        
        # Factor 4: Reranker scores (if available, need normalization)
        # Cross-encoder scores can be negative, normalize using sigmoid
        if reranker_scores:
            # Normalize reranker scores using sigmoid: 1 / (1 + e^(-x))
            import math
            normalized_scores = [1 / (1 + math.exp(-score)) for score in reranker_scores]
            avg_normalized_score = sum(normalized_scores) / len(normalized_scores)
            quality_factors.append(avg_normalized_score)
        
        # Calculate weighted average
        quality_score = sum(quality_factors) / len(quality_factors)
        
        return min(max(quality_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_enhanced_confidence(
        self,
        base_confidence: float,
        context_quality: float,
        response_length: int,
        context_length: int
    ) -> float:
        """
        Calculate enhanced confidence score combining multiple signals.
        
        Args:
            base_confidence: Base confidence from LLM
            context_quality: Quality of retrieved context
            response_length: Length of generated response
            context_length: Total length of context
            
        Returns:
            Enhanced confidence score (0-1)
        """
        # Start with base confidence
        confidence = base_confidence
        
        # Adjust based on context quality (less aggressive penalty)
        # Only penalize if context quality is very low
        if context_quality < 0.3:
            confidence *= 0.8
        elif context_quality < 0.5:
            confidence *= 0.9
        # Otherwise, keep base confidence
        
        # Adjust based on response coverage
        if context_length > 0:
            coverage_ratio = response_length / context_length
            # Penalize if response is too short (< 5% of context)
            if coverage_ratio < 0.05:
                confidence *= 0.8
            # Penalize if response is suspiciously long (> 80% of context)
            elif coverage_ratio > 0.8:
                confidence *= 0.9
        
        # Penalize very short responses (less than 30 chars)
        if response_length < 30:
            confidence *= 0.7
        
        return min(confidence, 1.0)
    
    def _generate_insufficient_context_response(self) -> str:
        """Generate response for insufficient context quality."""
        return (
            "I found some potentially relevant information, but I'm not confident "
            "it adequately addresses your question. Could you please rephrase or "
            "provide more specific details about what you're looking for?"
        )
    
    def _generate_confidence_feedback(
        self,
        confidence: float,
        context_quality: float,
        response: str,
        context: List[str]
    ) -> str:
        """
        Generate feedback for low confidence responses.
        
        Args:
            confidence: Enhanced confidence score
            context_quality: Context quality score
            response: Generated response
            context: Retrieved context
            
        Returns:
            Feedback string for retry
        """
        feedback_parts = [
            f"The previous response had low confidence ({confidence:.2f})."
        ]
        
        if context_quality < 0.5:
            feedback_parts.append(
                f"The retrieved context quality is moderate ({context_quality:.2f})."
            )
        
        if len(response) < 50:
            feedback_parts.append(
                "The response was too brief and lacked sufficient detail."
            )
        
        feedback_parts.append(
            "Please regenerate a more confident and detailed response that "
            "thoroughly addresses the question using the provided context."
        )
        
        return " ".join(feedback_parts)
    
    def _generate_grounding_feedback(
        self,
        grounding_score: float,
        response: str,
        context: List[str]
    ) -> str:
        """
        Generate feedback for poorly grounded responses.
        
        Args:
            grounding_score: Grounding score
            response: Generated response
            context: Retrieved context
            
        Returns:
            Feedback string for retry
        """
        return (
            f"The previous response was not sufficiently grounded in the provided context "
            f"(grounding score: {grounding_score:.2f}). The response contained information "
            f"that could not be verified against the retrieved documents. "
            f"Please regenerate the response using ONLY information explicitly stated in "
            f"the provided context. Do not add external knowledge or make assumptions."
        )
    
    def _generate_hallucination_feedback(self, indicators: List[str]) -> str:
        """
        Generate feedback for responses with hallucination indicators.
        
        Args:
            indicators: List of hallucination indicators
            
        Returns:
            Feedback string for retry
        """
        indicator_types = set()
        for indicator in indicators:
            if "unsupported" in indicator:
                indicator_types.add("unsupported claims")
            elif "number" in indicator:
                indicator_types.add("unverified numbers")
        
        indicator_str = ", ".join(indicator_types) if indicator_types else "unsupported information"
        
        return (
            f"The previous response contained potential hallucinations ({indicator_str}). "
            f"Please regenerate the response ensuring that ALL facts, numbers, and claims "
            f"are directly supported by the provided context. If information is not in the "
            f"context, explicitly state that it is not available."
        )


# Example usage
if __name__ == "__main__":
    guardrails = GuardrailsEngine()

    # Test case 1: Valid response
    response = "To reset your password, click the Forgot Password link."
    context = ["Password reset procedure: Click Forgot Password link below the login field."]
    is_valid, final_response, metadata = guardrails.validate(
        response, 0.9, context, "How do I reset my password?"
    )
    print(f"Test 1 - Valid: {is_valid}")
    print(f"Metadata: {metadata}\n")

    # Test case 2: Low confidence
    is_valid, final_response, metadata = guardrails.validate(
        response, 0.3, context, "How do I reset my password?"
    )
    print(f"Test 2 - Low confidence: {is_valid}")
    print(f"Response: {final_response}\n")

    # Test case 3: Hallucination
    hallucinated_response = "According to my knowledge, you need to contact support."
    is_valid, final_response, metadata = guardrails.validate(
        hallucinated_response, 0.9, context, "How do I reset my password?"
    )
    print(f"Test 3 - Hallucination: {is_valid}")
    print(f"Indicators: {metadata.get('indicators')}")
