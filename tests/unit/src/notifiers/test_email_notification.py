from unittest.mock import MagicMock, Mock, patch

from src.notifiers.base_notification import BasicNotification, NotificationComponent
from src.notifiers.email_notification import EmailConfig, EmailNotificationDecorator


class TestEmailNotificationDecorator:
    """Test cases for EmailNotificationDecorator class"""

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters provided"""
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            smtp_server="smtp.example.com",
            smtp_port=465,
            username="user@example.com",
            password="password123",
            from_addr="sender@example.com",
            subject="Test Subject",
        )

        assert decorator._component is component
        assert decorator.smtp_server == "smtp.example.com"
        assert decorator.smtp_port == 465
        assert decorator.username == "user@example.com"
        assert decorator.password == "password123"
        assert decorator.from_addr == "sender@example.com"
        assert decorator.subject == "Test Subject"

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters"""
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        assert decorator.smtp_server == "smtp.gmail.com"
        assert decorator.smtp_port == 587
        assert decorator.from_addr == "user@example.com"  # Defaults to username
        assert decorator.subject == "Notification"

    def test_from_addr_defaults_to_username(self):
        """Test that from_addr defaults to username when not provided"""
        component = BasicNotification()
        username = "user@example.com"
        decorator = EmailNotificationDecorator(
            component=component,
            username=username,
            password="password123",
        )

        assert decorator.from_addr == username

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_with_credentials_success(self, mock_smtp):
        """Test send method with credentials - successful email"""
        # Setup mock
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        # Create component and decorator
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        # Send notification
        message = "Test message"
        recipient = "recipient@example.com"
        result = decorator.send(message, recipient)

        # Verify SMTP calls
        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
        mock_server.ehlo.assert_called()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user@example.com", "password123")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

        assert result is True

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_with_credentials_failure(self, mock_smtp, capsys):
        """Test send method with credentials - failed email"""
        # Setup mock to raise exception
        mock_smtp.side_effect = Exception("SMTP connection failed")

        # Create component and decorator
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        # Send notification
        message = "Test message"
        recipient = "recipient@example.com"
        result = decorator.send(message, recipient)

        # Verify error was printed
        captured = capsys.readouterr()
        assert "Error sending email" in captured.out
        assert "SMTP connection failed" in captured.out

        # Result should be False because email failed (success and email_success = True and False = False)
        assert result is False

    def test_send_without_credentials(self, capsys):
        """Test send method without credentials"""
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username=None,
            password=None,
        )

        message = "Test message"
        recipient = "recipient@example.com"
        result = decorator.send(message, recipient)

        # Verify message about missing credentials
        captured = capsys.readouterr()
        assert "Email credentials not configured" in captured.out

        # Should still return True from basic notification
        assert result is True

    def test_send_without_username(self, capsys):
        """Test send method with password but no username"""
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username=None,
            password="password123",
        )

        result = decorator.send("Test", "test@example.com")

        captured = capsys.readouterr()
        assert "Email credentials not configured" in captured.out
        assert result is True

    def test_send_without_password(self, capsys):
        """Test send method with username but no password"""
        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password=None,
        )

        result = decorator.send("Test", "test@example.com")

        captured = capsys.readouterr()
        assert "Email credentials not configured" in captured.out
        assert result is True

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_delegates_to_parent(self, mock_smtp):
        """Test that send method calls parent component"""
        # Setup mock
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        # Create mock component
        mock_component = MagicMock(spec=NotificationComponent)
        mock_component.send.return_value = True

        decorator = EmailNotificationDecorator(
            component=mock_component,
            username="user@example.com",
            password="password123",
        )

        message = "Test message"
        recipient = "recipient@example.com"
        decorator.send(message, recipient)

        # Verify parent was called
        mock_component.send.assert_called_once_with(message, recipient)

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_email_creates_correct_message(self, mock_smtp):
        """Test that _send_email creates correct MIME message"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        component = BasicNotification()
        subject = "Custom Subject"
        from_addr = "sender@example.com"
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
            from_addr=from_addr,
            subject=subject,
        )

        message = "Test message content"
        recipient = "recipient@example.com"
        decorator.send(message, recipient)

        # Get the message that was sent
        call_args = mock_server.send_message.call_args
        sent_msg = call_args[0][0]

        assert sent_msg["Subject"] == subject
        assert sent_msg["From"] == from_addr
        assert sent_msg["To"] == recipient
        assert message in sent_msg.get_payload()

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_email_success_message(self, mock_smtp, capsys):
        """Test that successful email send prints success message"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        recipient = "recipient@example.com"
        decorator.send("Test", recipient)

        captured = capsys.readouterr()
        assert f"Email sent successfully to {recipient}" in captured.out

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_email_quit_called_in_finally(self, mock_smtp):
        """Test that server.quit() is called in finally block"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        decorator.send("Test", "test@example.com")

        # Verify quit was called
        mock_server.quit.assert_called_once()

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_email_quit_not_called_if_server_none(self, mock_smtp):
        """Test that quit is not called if server is None"""
        # Make SMTP constructor fail immediately
        mock_smtp.side_effect = Exception("Connection failed")

        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        # This should not raise an exception even though server is None
        result = decorator.send("Test", "test@example.com")
        # Result should be False because email failed
        assert result is False

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_send_email_exception_during_send_message(self, mock_smtp):
        """Test exception handling during send_message"""
        mock_server = Mock()
        mock_server.send_message.side_effect = Exception("Send failed")
        mock_smtp.return_value = mock_server

        component = BasicNotification()
        decorator = EmailNotificationDecorator(
            component=component,
            username="user@example.com",
            password="password123",
        )

        result = decorator.send("Test", "test@example.com")

        # quit should still be called in finally block
        mock_server.quit.assert_called_once()
        # Overall result should be False because email failed
        assert result is False

    @patch("src.notifiers.email_notification.smtplib.SMTP")
    def test_parent_failure_affects_result(self, mock_smtp):
        """Test that parent component failure affects overall result"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server

        # Create mock component that returns False
        mock_component = MagicMock(spec=NotificationComponent)
        mock_component.send.return_value = False

        decorator = EmailNotificationDecorator(
            component=mock_component,
            username="user@example.com",
            password="password123",
        )

        result = decorator.send("Test", "test@example.com")

        # Result should be False because parent returned False
        assert result is False


class TestEmailConfig:
    """Test cases for EmailConfig class"""

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters"""
        config = EmailConfig(
            smtp_server="smtp.example.com",
            smtp_port=465,
            username="user@example.com",
            password="password123",
            subject="Custom Subject",
        )

        assert config.smtp_server == "smtp.example.com"
        assert config.smtp_port == 465
        assert config.username == "user@example.com"
        assert config.password == "password123"
        assert config.subject == "Custom Subject"

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters"""
        config = EmailConfig()

        assert config.smtp_server == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.username is None
        assert config.password is None
        assert config.subject == "Notification"

    def test_initialization_partial_parameters(self):
        """Test initialization with some parameters"""
        config = EmailConfig(
            username="user@example.com",
            password="password123",
        )

        assert config.smtp_server == "smtp.gmail.com"  # Default
        assert config.smtp_port == 587  # Default
        assert config.username == "user@example.com"
        assert config.password == "password123"
        assert config.subject == "Notification"  # Default
