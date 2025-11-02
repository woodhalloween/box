"""
Test cases for _load_email_config() function to achieve 100% coverage.
"""

import configparser
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.video_processor import _load_email_config


class TestLoadEmailConfig:
    """Test suite for _load_email_config() function."""

    def test_create_config_ini_when_not_exists_with_env_vars(self, tmp_path, monkeypatch):
        """Test that config.ini is created with values from environment variables when it doesn't exist."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Set environment variables
            monkeypatch.setenv("EMAIL_USERNAME", "test@example.com")
            monkeypatch.setenv("EMAIL_PASSWORD", "testpass")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.example.com")
            monkeypatch.setenv("EMAIL_SMTP_PORT", "465")
            monkeypatch.setenv("EMAIL_SUBJECT", "Test Subject")
            monkeypatch.setenv("EMAIL_RECIPIENT", "recipient@example.com")

            # Call function
            result = _load_email_config()

            # Verify config.ini was created
            config_path = Path("config.ini")
            assert config_path.exists(), "config.ini should be created"

            # Verify file contents
            config = configparser.ConfigParser()
            config.read(config_path)
            assert config.has_section("email")
            assert config.get("email", "username") == "test@example.com"
            assert config.get("email", "password") == "testpass"
            assert config.get("email", "smtp_server") == "smtp.example.com"
            assert config.get("email", "smtp_port") == "465"
            assert config.get("email", "subject") == "Test Subject"
            assert config.get("email", "recipient") == "recipient@example.com"

            # Verify return values match env vars
            assert result["username"] == "test@example.com"
            assert result["password"] == "testpass"
            assert result["smtp_server"] == "smtp.example.com"
            assert result["smtp_port"] == "465"
            assert result["subject"] == "Test Subject"
            assert result["recipient"] == "recipient@example.com"
        finally:
            os.chdir(original_cwd)

    def test_create_config_ini_when_not_exists_without_env_vars(self, tmp_path, monkeypatch):
        """Test that config.ini is created with defaults when env vars are not set."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Ensure no env vars are set
            monkeypatch.delenv("EMAIL_USERNAME", raising=False)
            monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
            monkeypatch.delenv("EMAIL_SMTP_SERVER", raising=False)
            monkeypatch.delenv("EMAIL_SMTP_PORT", raising=False)
            monkeypatch.delenv("EMAIL_SUBJECT", raising=False)
            monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)

            # Call function
            result = _load_email_config()

            # Verify config.ini was created
            config_path = Path("config.ini")
            assert config_path.exists(), "config.ini should be created"

            # Verify file contents with defaults
            config = configparser.ConfigParser()
            config.read(config_path)
            assert config.has_section("email")
            assert config.get("email", "username") == ""
            assert config.get("email", "password") == ""
            assert config.get("email", "smtp_server") == "smtp.gmail.com"
            assert config.get("email", "smtp_port") == "587"
            assert config.get("email", "subject") == "Hand raise detected"
            assert config.get("email", "recipient") == ""

            # Verify return values use defaults
            assert result["username"] is None
            assert result["password"] is None
            assert result["smtp_server"] == "smtp.gmail.com"
            assert result["smtp_port"] == "587"
            assert result["subject"] == "Hand raise detected"
            assert result["recipient"] is None
        finally:
            os.chdir(original_cwd)

    def test_read_existing_config_ini_with_values(self, tmp_path, monkeypatch):
        """Test reading existing config.ini with all values set."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with values
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "config@example.com",
                "password": "configpass",
                "smtp_server": "smtp.config.com",
                "smtp_port": "993",
                "subject": "Config Subject",
                "recipient": "config_recipient@example.com",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set different env vars to ensure config.ini takes priority
            monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
            monkeypatch.setenv("EMAIL_PASSWORD", "envpass")

            # Call function
            result = _load_email_config()

            # Verify return values come from config.ini, not env vars
            assert result["username"] == "config@example.com"
            assert result["password"] == "configpass"
            assert result["smtp_server"] == "smtp.config.com"
            assert result["smtp_port"] == "993"
            assert result["subject"] == "Config Subject"
            assert result["recipient"] == "config_recipient@example.com"
        finally:
            os.chdir(original_cwd)

    def test_read_existing_config_ini_with_empty_values_fallback_to_env(self, tmp_path, monkeypatch):
        """Test that empty values in config.ini fall back to environment variables."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with empty values
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "",
                "password": "",
                "smtp_server": "",
                "smtp_port": "",
                "subject": "",
                "recipient": "",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set environment variables
            monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
            monkeypatch.setenv("EMAIL_PASSWORD", "envpass")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")
            monkeypatch.setenv("EMAIL_SMTP_PORT", "888")
            monkeypatch.setenv("EMAIL_SUBJECT", "Env Subject")
            monkeypatch.setenv("EMAIL_RECIPIENT", "env_recipient@example.com")

            # Call function
            result = _load_email_config()

            # Verify return values come from env vars when config.ini values are empty
            assert result["username"] == "env@example.com"
            assert result["password"] == "envpass"
            assert result["smtp_server"] == "smtp.env.com"
            assert result["smtp_port"] == "888"
            assert result["subject"] == "Env Subject"
            assert result["recipient"] == "env_recipient@example.com"
        finally:
            os.chdir(original_cwd)

    def test_read_existing_config_ini_with_empty_values_fallback_to_defaults(self, tmp_path, monkeypatch):
        """Test that empty values in config.ini fall back to defaults when env vars not set."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with empty values
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "",
                "password": "",
                "smtp_server": "",
                "smtp_port": "",
                "subject": "",
                "recipient": "",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Ensure no env vars are set
            monkeypatch.delenv("EMAIL_USERNAME", raising=False)
            monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
            monkeypatch.delenv("EMAIL_SMTP_SERVER", raising=False)
            monkeypatch.delenv("EMAIL_SMTP_PORT", raising=False)
            monkeypatch.delenv("EMAIL_SUBJECT", raising=False)
            monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)

            # Call function
            result = _load_email_config()

            # Verify return values use defaults
            assert result["username"] is None
            assert result["password"] is None
            assert result["smtp_server"] == "smtp.gmail.com"
            assert result["smtp_port"] == "587"
            assert result["subject"] == "Hand raise detected"
            assert result["recipient"] is None
        finally:
            os.chdir(original_cwd)

    def test_read_existing_config_ini_partial_values(self, tmp_path, monkeypatch):
        """Test reading config.ini with some values set and some missing."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with only some values
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "config@example.com",
                "password": "configpass",
                # smtp_server missing
                # smtp_port missing
                "subject": "Config Subject",
                # recipient missing
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set env vars for missing values
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")
            monkeypatch.setenv("EMAIL_SMTP_PORT", "888")
            monkeypatch.setenv("EMAIL_RECIPIENT", "env_recipient@example.com")

            # Call function
            result = _load_email_config()

            # Verify return values: config.ini values for set fields, env vars for missing fields
            assert result["username"] == "config@example.com"
            assert result["password"] == "configpass"
            assert result["smtp_server"] == "smtp.env.com"  # from env
            assert result["smtp_port"] == "888"  # from env
            assert result["subject"] == "Config Subject"
            assert result["recipient"] == "env_recipient@example.com"  # from env
        finally:
            os.chdir(original_cwd)

    def test_file_creation_failure(self, tmp_path, monkeypatch):
        """Test handling of file creation failure."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create a mock Path that will fail on open
            mock_path = Mock(spec=Path)
            mock_path.exists.return_value = False
            mock_path.open.side_effect = PermissionError("Cannot write to directory")
            mock_path.absolute.return_value = Path("/tmp/config.ini")

            # Set environment variables
            monkeypatch.setenv("EMAIL_USERNAME", "test@example.com")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.example.com")

            # Patch Path constructor to return our mock
            with patch("src.video_processor.Path", return_value=mock_path):
                # Call function - should not raise, but handle gracefully
                result = _load_email_config()

                # Should still return values from env vars despite write failure
                assert result["username"] == "test@example.com"
                assert result["smtp_server"] == "smtp.example.com"
        finally:
            os.chdir(original_cwd)

    def test_file_read_failure(self, tmp_path, monkeypatch):
        """Test handling of file read failure."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create a file that exists but will fail to read
            config_path = Path("config.ini")
            config_path.touch()

            # Mock configparser.ConfigParser.read to raise an exception
            with patch("configparser.ConfigParser.read", side_effect=Exception("Read error")):
                # Set environment variables as fallback
                monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
                monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")

                # Call function - should not raise, but handle gracefully
                result = _load_email_config()

                # Should fall back to env vars
                assert result["username"] == "env@example.com"
                assert result["smtp_server"] == "smtp.env.com"
        finally:
            os.chdir(original_cwd)

    def test_get_value_helper_exception_path(self, tmp_path, monkeypatch):
        """Test that get_value helper handles exceptions gracefully."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "test@example.com",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Mock config.has_section or config.has_option to raise exception
            with patch("configparser.ConfigParser.has_section", side_effect=Exception("Config error")):
                # Set environment variables as fallback
                monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")

                # Call function - should not raise, but handle gracefully
                result = _load_email_config()

                # Should fall back to env vars
                assert result["username"] == "env@example.com"
        finally:
            os.chdir(original_cwd)

    def test_get_value_helper_config_get_exception(self, tmp_path, monkeypatch):
        """Test that get_value helper handles config.get exceptions."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "test@example.com",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Mock config.get to raise exception for one key
            original_get = configparser.ConfigParser.get

            def mock_get(self, section, option, **kwargs):
                if option == "username":
                    raise Exception("Get error")
                return original_get(self, section, option, **kwargs)

            with patch("configparser.ConfigParser.get", mock_get):
                # Set environment variables as fallback
                monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")

                # Call function - should not raise, but handle gracefully
                result = _load_email_config()

                # Should fall back to env vars
                assert result["username"] == "env@example.com"
        finally:
            os.chdir(original_cwd)

    def test_config_section_not_exists(self, tmp_path, monkeypatch):
        """Test behavior when email section doesn't exist in config.ini."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini without email section
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["other_section"] = {
                "some_key": "some_value",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set environment variables
            monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")

            # Call function
            result = _load_email_config()

            # Should fall back to env vars
            assert result["username"] == "env@example.com"
            assert result["smtp_server"] == "smtp.env.com"
        finally:
            os.chdir(original_cwd)

    def test_config_option_not_exists(self, tmp_path, monkeypatch):
        """Test behavior when specific option doesn't exist in config.ini."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with email section but missing some options
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "config@example.com",
                # password missing
                # smtp_server missing
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set environment variables for missing options
            monkeypatch.setenv("EMAIL_PASSWORD", "envpass")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")

            # Call function
            result = _load_email_config()

            # Should use config.ini for existing options, env vars for missing
            assert result["username"] == "config@example.com"
            assert result["password"] == "envpass"
            assert result["smtp_server"] == "smtp.env.com"
        finally:
            os.chdir(original_cwd)

    def test_mixed_config_and_env_priority(self, tmp_path, monkeypatch):
        """Test that config.ini values take priority over env vars when both exist."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create config.ini with values
            config_path = Path("config.ini")
            config = configparser.ConfigParser()
            config["email"] = {
                "username": "config@example.com",
                "smtp_server": "smtp.config.com",
            }
            with config_path.open("w", encoding="utf-8") as f:
                config.write(f)

            # Set different env vars
            monkeypatch.setenv("EMAIL_USERNAME", "env@example.com")
            monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.env.com")

            # Call function
            result = _load_email_config()

            # Config.ini values should take priority
            assert result["username"] == "config@example.com"
            assert result["smtp_server"] == "smtp.config.com"
        finally:
            os.chdir(original_cwd)
