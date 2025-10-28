import smtplib
from email.mime.text import MIMEText

from .base_notification import BaseNotificationDecorator, NotificationComponent


# Email notification decorator
class EmailNotificationDecorator(BaseNotificationDecorator):
    def __init__(
        self,
        component: NotificationComponent,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        from_addr: str = None,
        subject: str = "Notification",
    ):
        super().__init__(component)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr or username
        self.subject = subject

    def send(self, message: str, recipient: str) -> bool:
        # First, call the parent component's send method
        success = super().send(message, recipient)

        # Then add email functionality
        if self.username and self.password:
            email_success = self._send_email(message, recipient)
            return success and email_success
        print("Email credentials not configured, skipping email notification")
        return success

    def _send_email(self, message: str, recipient: str) -> bool:
        """Send email using SMTP"""
        server = None
        try:
            # Create the email
            msg = MIMEText(message)
            msg["Subject"] = self.subject
            msg["From"] = self.from_addr
            msg["To"] = recipient

            # Connect and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.username, self.password)
            server.send_message(msg)

            print(f"Email sent successfully to {recipient}")
            return True

        except Exception as e:
            print(f"Error sending email to {recipient}: {e}")
            return False

        finally:
            if server is not None:
                server.quit()


# Example usage and configuration helper
class EmailConfig:
    """Configuration class for email settings"""

    def __init__(
        self, smtp_server="smtp.gmail.com", smtp_port=587, username=None, password=None, subject="Notification"
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.subject = subject
