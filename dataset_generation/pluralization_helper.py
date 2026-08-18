"""
This script helps with handling pluralization during the rule generation phase. 
We provide a pluralizing logic for English nouns following standard grammar rules
and a set of irregular nouns that occur in MoRe's taxonomy.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Tuple, Generator, Optional, Callable
import json
import csv


# here we manually include the plural forms of certain irregular nouns present in the taxonomy

_IRREGULARS = {
    "mouse": "mice", "louse": "lice", "goose": "geese",
    "man": "men", "woman": "women", "child": "children",
    "tooth": "teeth", "foot": "feet", "person": "people",
    "human": "humans", "ox": "oxen", "wolf": "wolves",
    "calf": "calves", "knife": "knives","shelf": "shelves", 
    "loaf": "loaves", "leaf": "leaves", "fish": "fish", "sheep": "sheep", 
    "deer": "deer", "bison": "bison", "moose": "moose", "salmon": "salmon",
    "trout": "trout", "cod": "cod", "bacterium": "bacteria", "bacteria": "bacteria",
    "protozoan": "protozoa", "protozoa": "protozoa", "fungus": "fungi",
    "microbe": "microbes", "virus": "viruses","cactus": "cacti", "nucleus": "nuclei"
}


_VOWELS = set("aeiou")

def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def pluralize(word: str) -> str:
    
    """ Returns the plural form of an english noun."""
    
    w = (word or "").strip()
    if not w:
        return w
    lw = w.lower()
    if lw.endswith("s") and lw not in {"is", "has"} and lw not in _IRREGULARS:
        return w
    if lw in _IRREGULARS:
        return _IRREGULARS[lw]
    if lw.endswith(("craft", "series", "species", "aircraft")):
        return w
    if lw.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if lw.endswith("y") and len(lw) > 1 and not _is_vowel(lw[-2]):
        return w[:-1] + "ies"
    if lw.endswith("fe"):
        return w[:-2] + "ves"
    if lw.endswith("f"):
        return w[:-1] + "ves"
    return w + "s"
