import os, sys
def _build_domain_ids():
    business_type_list = ['shein','shopee_cps','aecps','aerta','everglowly','lazada_rta','lazada_cps','ttshop','miravia_rta','liongame','cyberclickshecurve','detroitrain','swiftlink','saker','starpony','alibaba','mena_ae_cps']
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    label_encoder = LabelEncoder()
    label_encoder.fit(business_type_list)

    business_types_str = business_type_list
    domain_ids_int = label_encoder.transform(business_types_str)
    print("Business Types (str):", business_types_str)
    print("Domain IDs (int):", domain_ids_int.tolist()) # [0, 2, 1, 0, 3]

    # 在反向转换时
    original_types = label_encoder.inverse_transform(domain_ids_int)
    print("Back to str:", original_types.tolist())

_build_domain_ids()