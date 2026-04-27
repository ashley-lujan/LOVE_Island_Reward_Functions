import smtplib, traceback
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