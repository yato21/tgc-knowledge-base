# Вендоризованное подмножество snowballstemmer 3.1.1 (только russian + english).
# Источник: https://pypi.org/project/snowballstemmer/ — BSD-3-Clause (см. LICENSE).
# Вендоризация по решению task-9 (Софья, §4.3): pip install из песочницы агентов
# не исполним — стемминг должен работать с любым python3 без зависимостей.
from .russian_stemmer import RussianStemmer
from .english_stemmer import EnglishStemmer

_stemmers = {}


def stemmer(name):
    if name not in _stemmers:
        _stemmers[name] = {"russian": RussianStemmer(), "english": EnglishStemmer()}[name]
    return _stemmers[name]
