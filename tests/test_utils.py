from pusoy.utils import Card, Suit, Value, string_to_card
import pytest

def test_suit():
    assert Suit.clubs < Suit.spades
    assert Suit.spades < Suit.hearts
    assert Suit.hearts < Suit.diamonds
    assert Suit.diamonds == Suit.diamonds

def test_str_to_card():
    assert string_to_card('2H') == Card(Value.two, Suit.hearts)
    assert string_to_card('10D') == Card(Value.ten, Suit.diamonds)
    assert string_to_card('JC') == Card(Value.jack, Suit.clubs)