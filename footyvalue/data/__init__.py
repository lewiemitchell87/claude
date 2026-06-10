"""Data adapters: historical results and live odds.

These are deliberately thin and swappable. The engine only needs:

* a list of :class:`footyvalue.ratings.Match` to fit ratings, and
* a normalised odds structure ``{market: {selection: decimal_odds}}`` to price.

Wire in any source you like by producing those shapes.
"""
