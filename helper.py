import requests


def confirm_checkin(user_id, face):
    payload = {
        "userId": user_id,
        "face": face
    }

    try:
        r = requests.post("http://localhost:4000/checkin/create", data=payload)
    except:
        pass

    return True
