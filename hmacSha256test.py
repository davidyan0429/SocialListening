# encoding=utf-8
import base64
import hashlib
import hmac
import pytz

from datetime import datetime
import requests

# 1. 计算sign
method = "GET"
uri = "/session"
appId = "TpBXTw0WC6Zz"
tz = pytz.timezone('utc')
timestamp = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%SZ")  # timestamp 时间为0时区时间
signature_method = "HmacSHA256"
signature_version = "1"
query = ""

appSecret = "b99bt5TnNQNG7BZcfrgCgC4UqHfFhBhh"
reqInfo = "\n".join([method, uri, appId, timestamp, signature_method, signature_version, query])

sign = base64.b64encode(hmac.new(appSecret, msg=reqInfo, digestmod=hashlib.sha256).digest())

# 2. 请求api /session
url = "http://report.socialmaster.com.cn/api/session"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Cache-Control": "no-cache",
    "X-Auth-Signature": sign,
    "X-Auth-Key": appId,
    "X-Auth-Timestamp": timestamp,
    "X-Auth-Sign-Method": signature_method,
    "X-Auth-Sign-Version": signature_version
}
r = requests.get(url, headers=headers)
print(r.json())