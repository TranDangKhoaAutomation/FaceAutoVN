import os
import requests
tokenEmail = os.getenv("TEMPMAIL_TOKEN", "")
def get_temp_email():
    global tokenEmail
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + tokenEmail,
    }
    json_data = {'user': '', 'domain': 'tempmail.ckvn.edu.vn'}
    res = requests.post('https://tempmail.id.vn/api/email/create', headers=headers, json=json_data)
    print(res.json())
    data = res.json()
    if data["success"] is True:
        print(f"🧠 Đã tạo email tạm thời: {data['data']['email']}")
        return {"id": data["data"]["id"], "email": data["data"]["email"]}
    else:
        print("Lỗi khi tạo email tạm:" + data["message"])
        exit(0)

def listEmail():
    global tokenEmail
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + tokenEmail,
    }

    response = requests.get('https://tempmail.id.vn/api/email', headers=headers)
    return response.json()

def read_email(mail_id):
    global tokenEmail
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + tokenEmail,
    }
    res = requests.get(f"https://tempmail.id.vn/api/email/{mail_id}", headers=headers)
    res.raise_for_status()
    return res.json()