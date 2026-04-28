import os
import smtplib
import traceback
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def setup_alerts(alert_cfg):
    """Validate alert config on startup so you catch issues early."""
    if not alert_cfg.gmail_user or not alert_cfg.gmail_password:
        raise ValueError("Alert config missing gmail credentials")

def send_alert(cfg, subject, body):
    """Send alert using credentials from Hydra cfg."""
    if not cfg.alerts.enabled:
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"[CHPC Eureka] {subject}"
    msg["From"] = cfg.alerts.gmail_user
    msg["To"] = cfg.alerts.recipient

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg.attach(MIMEText(f"{body}\n\n---\nSent: {timestamp}", "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(cfg.alerts.gmail_user, cfg.alerts.gmail_password)
        server.sendmail(cfg.alerts.gmail_user, cfg.alerts.recipient, msg.as_string())
    
    print(f"[Alert sent] {subject}")


def send_alert_with_attachments(cfg, subject: str, body: str, attachments: list) -> None:
    """Send email with file attachments (e.g., GIF files)."""
    if not cfg.alerts.enabled:
        return

    msg = MIMEMultipart()
    msg["Subject"] = f"[CHPC LOVE Island] {subject}"
    msg["From"] = cfg.alerts.gmail_user
    msg["To"] = cfg.alerts.recipient

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg.attach(MIMEText(f"{body}\n\n---\nSent: {timestamp}", "plain"))

    for path in attachments:
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(cfg.alerts.gmail_user, cfg.alerts.gmail_password)
        server.sendmail(cfg.alerts.gmail_user, cfg.alerts.recipient, msg.as_string())

    print(f"[Alert sent with {len(attachments)} attachment(s)] {subject}")