import uuid

if __name__ == '__main__':
    random_uuid = uuid.uuid4().hex[:6]
    print(random_uuid)