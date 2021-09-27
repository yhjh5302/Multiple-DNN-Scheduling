import database
import os
import yaml
import copy
import shutil


# connect with algorithm
def make_app_name(svc_name, msvc_name):
    return "svc-%s-msvc-%s" % (svc_name, msvc_name)


conn = database.MyConnector()
msvc_info, require_info, svc_map = conn.get_msvc_data()
svc_info = conn.get_svc_data()
node_info, resource_info = conn.get_node_info()

# todo create msvc yaml script
SVC_YAML_FOLDER = "./svc_yaml"
YAML_TEMPLATE = "pod_script/template.yaml"

with open(YAML_TEMPLATE, 'r') as rf:
    yaml_template = yaml.safe_load_all(rf)
    yaml_template = list(yaml_template)

for svc_idx in svc_map:
    folder_path = os.path.join(SVC_YAML_FOLDER, "svc-%s" % svc_idx)
    if os.path.isdir(folder_path):
        # os.rmdir(folder_path)
        shutil.rmtree(folder_path)
    elif os.path.isfile(folder_path):
        os.remove(folder_path)

    os.makedirs(folder_path, exist_ok=True)
    for msvc_idx in svc_map[svc_idx]:
        path = os.path.join(folder_path, "msvc-%s.yaml" % msvc_idx)
        data = copy.deepcopy(yaml_template)

        app_name = make_app_name(svc_idx, msvc_idx)

        data[0]['metadata']['name'] = "svc-" + str(msvc_info[msvc_idx][0]) + "-" + str(msvc_idx)
        data[0]['metadata']['labels']['app'] = app_name

        data[0]['spec']['selector']['matchLabels']['app'] = app_name
        data[0]['spec']['template']['metadata']['labels']['app'] = app_name

        if msvc_idx != svc_map[svc_idx][-1]:
            next_addr = "http://test-" + make_app_name(svc_idx, msvc_idx + 1) + ":4000"
        else:
            next_addr = "end"

        data[0]['spec']['template']['spec']['containers'][0]['args'] = [
            "--fog_sleep_time", str(msvc_info[msvc_idx][1]),
            "--cloud_sleep_time", str(msvc_info[msvc_idx][2]),
            "--cloud_trans_time", str(svc_info[svc_idx][2]),
            "--is_fog", str(True),
            "--next_address", next_addr
        ]

        data[1]['metadata']['name'] = "test-" + app_name
        data[1]['spec']['selector']['app'] = app_name

        with open(path, 'w') as wf:
            yaml.safe_dump_all(data, wf)

