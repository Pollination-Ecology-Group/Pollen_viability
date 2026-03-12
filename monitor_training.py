import os
import time
import subprocess
import smtplib
from email.message import EmailMessage

# --- CONFIGURATION ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "jakubstenc@gmail.com"
RECEIVER_EMAIL = "jakubstenc@gmail.com"

# Read password from environment
# You need to generate an App Password in your Google Account:
# https://myaccount.google.com/apppasswords
SMTP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

NAMESPACE = "stenc-ns"
JOB_NAME = "pollen-train-job"


def send_email(subject, body):
    if not SMTP_PASSWORD:
        print("❌ Cannot send email: GMAIL_APP_PASSWORD is not set.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        print(f"📧 Sending email via {SMTP_SERVER}...")
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def check_job_status():
    """Checks the k8s job status and returns (is_done, status_string)"""
    try:
        # Check for success
        output = subprocess.check_output(
            f'export KUBECONFIG="$(pwd)/kubeconfig.yaml"; ./kubectl get job {JOB_NAME} -n {NAMESPACE} -o jsonpath="{{.status.succeeded}}"',
            shell=True,
            text=True
        ).strip()
        if output == "1":
            return True, "Succeeded"
        
        # Check for failure
        output_fail = subprocess.check_output(
            f'export KUBECONFIG="$(pwd)/kubeconfig.yaml"; ./kubectl get job {JOB_NAME} -n {NAMESPACE} -o jsonpath="{{.status.failed}}"',
            shell=True,
            text=True
        ).strip()
        if output_fail and int(output_fail) > 0:
            return True, "Failed"
            
    except subprocess.CalledProcessError:
        print("⚠️ Warning: Could not reach Kubernetes cluster to check job.")
        
    return False, "Running"


def main():
    if not SMTP_PASSWORD:
        print("\n" + "="*50)
        print("⚠️ WARNING: Email password not configured!")
        print("You must export GMAIL_APP_PASSWORD=your_password before running this script.")
        print("Get an App Password here: https://myaccount.google.com/apppasswords")
        print("="*50 + "\n")
        return

    print(f"👀 Monitoring Kubernetes job '{JOB_NAME}'...")
    print(f"📧 Notifications will be sent to {RECEIVER_EMAIL}")

    while True:
        is_done, status = check_job_status()
        
        if is_done:
            print(f"\n🎉 Job finished with status: {status}")
            subject = f"Pollen Training Job: {status}"
            body = f"Hello,\n\nYour Kubernetes training job '{JOB_NAME}' has finished with status '{status}'.\n\nIf it succeeded, your model weights should now be syncing directly to your S3 bucket.\n\nBest regards,\nYour automated monitor"
            
            send_email(subject, body)
            break
            
        print(".", end="", flush=True)
        time.sleep(60) # Check every 60 seconds

if __name__ == "__main__":
    main()
