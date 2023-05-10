import os

def batch_process_files(root_dir):
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)
        if os.path.isdir(file_path):
            protein_path = os.path.join(file_path,f"{filename}_protein.pdb")
            out_path = os.path.join(file_path,"ANM")
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            os.system(
                f"prody anm {protein_path} "
                f"-a -o {out_path} "
                "-R"
            )
            print(f"Processing {filename}")
        else:
            process_file(file_path)

def process_file(file_path):
    print(f"This isn't a file: {file_path}")

if __name__ == "__main__":
    # 调用 batch_process_files() 函数以对根目录中的所有文件进行处理
    root_dir = '/data4/yxzhang/DEGNA/DATASET/general-set-except-refined'
    # root_dir="/data4/yxzhang/EGNA/DATASET/Test2"
    batch_process_files(root_dir)
