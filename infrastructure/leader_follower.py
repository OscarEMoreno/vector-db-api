
from infrastructure.repositories import LibraryRepository
from typing import List, Optional


class LeaderFollowerRepository(LibraryRepository):
    def __init__(self, leader: LibraryRepository, followers: List[LibraryRepository]):
        self.leader = leader
        self.followers = followers

    def add(self, lib: LibraryRepository) -> None:
        self.leader.add(lib)
        for f in self.followers:
            f.add(lib)

    def get(self, lib_id: str) -> Optional[LibraryRepository]:
        return self.leader.get(lib_id)

    def update(self, lib: LibraryRepository) -> None:
        self.leader.update(lib)
        for f in self.followers:
            f.update(lib)

    def delete(self, lib_id: str) -> None:
        self.leader.delete(lib_id)
        for f in self.followers:
            f.delete(lib_id)

    def list_all(self) -> List[LibraryRepository]:
        return self.leader.list_all()
