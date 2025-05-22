from typing import List, Optional
from infrastructure.repositories import BaseLibraryRepository


class LeaderFollowerRepository(BaseLibraryRepository):
    def __init__(self, leader: BaseLibraryRepository, followers: List[BaseLibraryRepository]):
        self.leader = leader
        self.followers = followers

    def add(self, lib: BaseLibraryRepository) -> None:
        self.leader.add(lib)
        for f in self.followers:
            f.add(lib)

    def get(self, lib_id: str) -> Optional[BaseLibraryRepository]:
        return self.leader.get(lib_id)

    def update(self, lib: BaseLibraryRepository) -> None:
        self.leader.update(lib)
        for f in self.followers:
            f.update(lib)

    def delete(self, lib_id: str) -> None:
        self.leader.delete(lib_id)
        for f in self.followers:
            f.delete(lib_id)

    def list_all(self) -> List[BaseLibraryRepository]:
        return self.leader.list_all()
