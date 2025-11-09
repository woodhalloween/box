from unittest.mock import MagicMock

import pytest

from src.notifiers.base_notification import (
    BaseNotificationDecorator,
    BasicNotification,
    NotificationComponent,
)


class TestBasicNotification:
    """Test cases for BasicNotification class"""

    def test_send_returns_true(self):
        """Test that send method returns True"""
        notification = BasicNotification()
        result = notification.send("Test message", "test@example.com")
        assert result is True

    def test_send_prints_message(self, capsys):
        """Test that send method prints the correct message"""
        notification = BasicNotification()
        message = "Test message"
        recipient = "test@example.com"
        notification.send(message, recipient)

        captured = capsys.readouterr()
        assert f"Basic notification: {message} to {recipient}" in captured.out


class TestBaseNotificationDecorator:
    """Test cases for BaseNotificationDecorator class"""

    def test_initialization_stores_component(self):
        """Test that decorator stores the wrapped component"""
        component = BasicNotification()
        decorator = BaseNotificationDecorator(component)
        assert decorator._component is component

    def test_send_delegates_to_component(self):
        """Test that send method delegates to wrapped component"""
        # Create a mock component
        mock_component = MagicMock(spec=NotificationComponent)
        mock_component.send.return_value = True

        # Create decorator with mock component
        decorator = BaseNotificationDecorator(mock_component)

        # Call send
        message = "Test message"
        recipient = "test@example.com"
        result = decorator.send(message, recipient)

        # Verify delegation
        mock_component.send.assert_called_once_with(message, recipient)
        assert result is True

    def test_send_returns_component_result(self):
        """Test that send method returns the result from wrapped component"""
        # Create a mock component that returns False
        mock_component = MagicMock(spec=NotificationComponent)
        mock_component.send.return_value = False

        decorator = BaseNotificationDecorator(mock_component)
        result = decorator.send("Test", "test@example.com")

        assert result is False


class TestNotificationComponent:
    """Test cases for NotificationComponent abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that NotificationComponent cannot be instantiated directly"""
        with pytest.raises(TypeError):
            NotificationComponent()

    def test_subclass_must_implement_send(self):
        """Test that subclass must implement send method"""

        class IncompleteNotification(NotificationComponent):
            pass

        with pytest.raises(TypeError):
            IncompleteNotification()

    def test_subclass_with_send_can_instantiate(self):
        """Test that proper subclass can be instantiated"""

        class CompleteNotification(NotificationComponent):
            def send(self, message: str, recipient: str) -> bool:
                return True

        notification = CompleteNotification()
        assert isinstance(notification, NotificationComponent)

    def test_abstract_method_pass_statement(self):
        """Test to cover the pass statement in abstract method"""

        class TestNotification(NotificationComponent):
            def send(self, message: str, recipient: str) -> bool:
                # Call the abstract method's pass statement via super
                super().send(message, recipient)
                return True

        notification = TestNotification()
        result = notification.send("test", "test@example.com")
        assert result is True
