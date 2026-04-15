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
        query: str
    ) -> Tuple[bool, str, Dict]:
        """
        Validate response against all guardrails.

        Args:
            response: Generated response text
            confidence: Response confidence score
            retrieved_context: Context used for generation
            query: Original user query

        Returns:
            Tuple of (is_valid, final_response, validation_metadata)
        """
        logger.info("Running guardrails validation...")

        validation_results = {
            "confidence_check": False,
            "grounding_check": False,
            "hallucination_check": False,
            "safety_check": False,
            "pii_check": False
        }

        # 1. Confidence check
        if confidence < self.confidence_threshold:
            logger.warning(f"Low confidence: {confidence:.2f}")
            return False, self._generate_low_confidence_response(), {
                **validation_results,
                "failure_reason": "low_confidence",
                "confidence": confidence
            }
        validation_results["confidence_check"] = True

        # 2. Context availability check
        if not retrieved_context or len(retrieved_context) == 0:
            logger.warning("No context available")
            return False, self._generate_no_context_response(), {
                **validation_results,
                "failure_reason": "no_context"
            }

        # 3. Grounding check
        grounding_score = self._calculate_grounding_score(response, retrieved_context)
        if grounding_score < self.grounding_threshold:
            logger.warning(f"Low grounding score: {grounding_score:.2f}")
            return False, self._generate_ungrounded_response(), {
                **validation_results,
                "failure_reason": "insufficient_grounding",
                "grounding_score": grounding_score
            }
        validation_results["grounding_check"] = True

        # 4. Hallucination detection
        has_hallucination, hallucination_indicators = self._detect_hallucination(
            response, retrieved_context
        )
        if has_hallucination:
            logger.warning(f"Hallucination detected: {hallucination_indicators}")
            return False, self._generate_safe_fallback(retrieved_context), {
                **validation_results,
                "failure_reason": "hallucination_detected",
                "indicators": hallucination_indicators
            }
        validation_results["hallucination_check"] = True

        # 5. Safety check
        is_safe, safety_issues = self._check_safety(response)
        if not is_safe:
            logger.warning(f"Safety issues detected: {safety_issues}")
            return False, "I cannot provide that information.", {
                **validation_results,
                "failure_reason": "safety_violation",
                "issues": safety_issues
            }
        validation_results["safety_check"] = True

        # 6. PII detection and filtering
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
            "confidence": confidence,
            "passed": True
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
