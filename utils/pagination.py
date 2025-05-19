from typing import List, TypeVar

T = TypeVar('T')


def paginate(items: List[T], offset: int = 0, limit: int = 100) -> List[T]:
    return items[offset: offset + limit]
