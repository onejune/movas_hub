import requests
dag_id = "ym_dsp_cps_check_union_feature_manal"
url = f"http://af.yeahmobi.com/api/v1/dags/{dag_id}/dagRuns"

payload = { "conf": {
    "pday": "2025-02-07",
    "phour": "00",
    "abtestkey": "yeahdsp_union_ftrl_purchase_huf_self_attr_v1_shein_stat"
} }
headers = {"Authorization": "Basic YWRtaW46YWRtaW4="}

response = requests.post(url, json=payload, headers=headers)

print(response.json())


