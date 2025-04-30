import smtplib
from email.mime.text import MIMEText

def send_alert():
    sender = "youremail@example.com"
    password = "yourpassword"
    receiver = "alertreceiver@example.com"
    
    msg = MIMEText("Unknown face detected!")
    msg['Subject'] = "Face Alert"
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("Alert email sent.")
    except Exception as e:
        print("Email failed:", e)