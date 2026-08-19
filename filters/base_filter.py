from ABC import ABC, abstractmethod 


class BaseFilter(ABC):

        
    @abstractmethod
    def apply(self, frame, face_landmarks):
        pass
    