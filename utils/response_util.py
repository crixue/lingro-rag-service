def server_response(business_code: int = 0, msg: str = 'success', data=None):
    response = {
        'code': business_code,
        'msg': msg,
        'data': data
    }
    return response
