import os


def create_test_log_dir(test_logs_dir, model:str):
    if "/models/" in model:
        model = model.split("/models/")[1]
    model = model.replace('/', '-')

    index = 1
    path = f"test{index}-{model}"
    full_path = os.path.join(test_logs_dir, path)
    while os.path.exists(full_path):
        path = f"test{index}-{model}"
        full_path = os.path.join(test_logs_dir, path)
        index += 1

    os.mkdir(full_path)
    return full_path


def create_log_file(log_dir, file_name: str, vuln):
    if ".json" in file_name:
        file_name = file_name.split(".json")[0]
    log_file = f"{file_name}-{vuln}"

    return os.path.join(log_dir,log_file)


def write_log(log_file: str, response: str, type):
    response = f"\n{'-'*50}{type}{'-'*50}{response}\n"
    with open(log_file,"a",encoding="utf-8") as f:
        f.write(response)

def write_log_summary(log_dir, text):
    log_file = os.path.join(log_dir, "log_summary.txt")
    print(text)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text+'\n')