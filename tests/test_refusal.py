"""Smoke tests for refusal detection."""
import pytest

from post_training_toolkit.refusal import (
    RefusalDetector,
    RefusalResult,
    RefusalType,
    is_refusal,
    detect_refusal,
    REFUSAL_PATTERNS,
)


class TestRefusalResult:
    """Tests for RefusalResult dataclass."""
    
    def test_refusal_result_creation(self):
        result = RefusalResult(
            is_refusal=True,
            refusal_type=RefusalType.SAFETY,
            confidence=0.95,
            matched_pattern="I cannot",
        )
        assert result.is_refusal == True
        assert result.refusal_type == RefusalType.SAFETY
        assert result.confidence == 0.95
        assert result.matched_pattern == "I cannot"


class TestRefusalDetection:
    """Tests for refusal detection functions."""
    
    @pytest.fixture
    def detector(self):
        return RefusalDetector()
    
    def test_clear_refusal_detected(self, detector):
        """Clear refusals should be detected."""
        refusal_texts = [
            "I cannot help with that request because it violates my guidelines.",
            "I'm sorry, but I can't assist with that type of content.",
            "I'm not able to provide that information as it would be harmful.",
            "I apologize, but I cannot engage with that topic due to safety concerns.",
        ]
        
        for text in refusal_texts:
            result = detector.detect(text)
            assert result.is_refusal, f"Should detect refusal in: {text}"
    
    def test_normal_response_not_refusal(self, detector):
        """Normal responses should not be flagged as refusals."""
        normal_texts = [
            "Here's how to make a cake: First, preheat the oven...",
            "The capital of France is Paris.",
            "def fibonacci(n):\n    if n <= 1:\n        return n",
            "Machine learning is a branch of artificial intelligence.",
            "To solve this problem, we can use dynamic programming.",
        ]
        
        for text in normal_texts:
            result = detector.detect(text)
            assert not result.is_refusal, f"Should NOT detect refusal in: {text}"
    
    def test_edge_cases(self, detector):
        """Edge cases should be handled correctly."""
        # Empty string
        result = detector.detect("")
        assert not result.is_refusal
        
        # Very short response
        result = detector.detect("Yes")
        assert not result.is_refusal
        
        # Response with "I can" (not "I cannot")
        result = detector.detect("I can definitely help you with that!")
        assert not result.is_refusal
    
    def test_is_refusal_convenience_function(self):
        """is_refusal should be a simple boolean check."""
        assert is_refusal("I cannot help with that.") == True
        assert is_refusal("Here's the answer you requested.") == False
    
    def test_detect_refusal_convenience_function(self):
        """detect_refusal should return a RefusalResult."""
        result = detect_refusal("I'm sorry, but I cannot assist with that.")
        assert isinstance(result, RefusalResult)
        assert result.is_refusal == True


class TestRefusalTypes:
    """Tests for different refusal type classification."""
    
    @pytest.fixture
    def detector(self):
        return RefusalDetector()
    
    def test_ethical_refusal_classification(self, detector):
        """Ethical refusals should be classified correctly."""
        text = "I cannot help with anything illegal or harmful."
        result = detector.detect(text)
        assert result.is_refusal
        # Should be classified as ethical refusal
    
    def test_capability_refusal_classification(self, detector):
        """Capability refusals should be classified correctly."""
        text = "I don't have the ability to browse the internet."
        result = detector.detect(text)
        assert result.is_refusal
        # Should be classified as capability refusal
    
    def test_ambiguous_refusal(self, detector):
        """Ambiguous cases should still be handled."""
        text = "I'm not sure I can do that."
        result = detector.detect(text)
        # May or may not be classified as refusal depending on threshold


class TestCustomPatterns:
    """Tests for custom refusal pattern configuration."""
    
    def test_custom_templates(self):
        """Custom templates should be usable."""
        custom_templates = [
            "declined",
            "request_denied",
        ]
        detector = RefusalDetector(templates=custom_templates)
        
        result = detector.detect("DECLINED: Cannot process this request.")
        assert result.is_refusal
    
    def test_custom_patterns_only(self):
        """Should be able to use only custom patterns."""
        # Empty templates and custom patterns
        detector = RefusalDetector(
            templates=[],
            patterns=[(r"^NOPE\s+", RefusalType.EXPLICIT)],
            min_length=5,  # Lower min_length for short test strings
        )
        
        # Default pattern should not match (templates are empty)
        result = detector.detect("I cannot help with that request.")
        assert not result.is_refusal
        
        # Custom pattern should match
        result = detector.detect("NOPE sorry")
        assert result.is_refusal


class TestDefaultPatterns:
    """Tests that default patterns cover common cases."""
    
    def test_default_patterns_comprehensive(self):
        """Default patterns should cover common refusal phrases."""
        common_refusals = [
            "I cannot help with that request.",
            "I can't assist with this type of content.",
            "I'm unable to provide that information.",
            "I am not able to do that for you.",
            "I'm sorry, but I cannot help you.",
            "I apologize, but I can't assist.",
            "I will not provide that information.",
            "I won't be able to help with this.",
        ]
        
        detector = RefusalDetector()
        for phrase in common_refusals:
            result = detector.detect(phrase)
            assert result.is_refusal, f"Default patterns should catch: {phrase}"
