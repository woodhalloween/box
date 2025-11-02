from abc import ABC, abstractmethod


# Base component interface
class NotificationComponent(ABC):
    @abstractmethod
    def send(self, message: str, recipient: str) -> bool:
        pass


# Concrete component - basic notification
class BasicNotification(NotificationComponent):
    def send(self, message: str, recipient: str) -> bool:
        print(f"Basic notification: {message} to {recipient}")
        return True


# Base decorator
class BaseNotificationDecorator(NotificationComponent):
    def __init__(self, component: NotificationComponent):
        self._component = component

    def send(self, message: str, recipient: str) -> bool:
        # Delegate to the wrapped component
        return self._component.send(message, recipient)
