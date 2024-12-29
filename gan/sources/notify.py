import os
import json
import requests
import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
import base64
from email.mime.text import MIMEText

class notifier:
    """
    Class to send notifications to phone and email
    """
    def __init__(self) -> None:
        self.users_key = [None]
        self.mail_address = [None]
        self.mailing = True
        try:
            self.scopes = ["https://www.googleapis.com/auth/gmail.send"]
            self.flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.scopes)
            self.creds = self.flow.run_local_server(port=0)
            self.service = build('gmail', 'v1', credentials=self.creds)
        except Exception as e:
            self.mailing = False
            print('Error loading gmail credentials')
            print(e)

    def send_email(self, subject: str, description: str) -> None:
        if self.mail_address[0] is None:
            return
        if not self.mailing:
            return
        for target in self.mail_address:
            message = MIMEText(description)
            message['to'] = target
            message['subject'] = subject
            create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
            try:
                # Send the email using the Gmail API
                response = self.service.users().messages().send(userId='me', body=create_message).execute()
                print('Email sent successfully')
            except Exception as e:
                print(f'Error sending email: {e}')

    def notify_phone(self, message: str, description: str) -> None:
        if self.users_key[0] is None:
            return
        for key in self.users_key:
            requests.post('https://api.mynotifier.app', {
                "apiKey": key,
                "message": message,
                "description": description,
                "type": "info", # info, error, warning or success
            })