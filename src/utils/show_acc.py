import os
import argparse
import tabulate

parser = argparse.ArgumentParser(description='Show Accuracy')
parser.add_argument('-i', '--input_dir', help="input dir", required=True)

args = parser.parse_args()
input_dir = args.input_dir


def show_acc():
    dirs = os.listdir(input_dir)

    res_list = []
    for sub_dir in dirs:
        if sub_dir.startswith("test_"):
            test_iter = sub_dir.split("_")[-1]
            res_map = {"iter": test_iter}

            eval_file = os.path.join(os.path.join(input_dir, sub_dir), "eval_mmau_mini.txt")
            if not os.path.exists(eval_file):
                continue
            with open(eval_file, "r", encoding="utf8") as reader:
                for line  in reader:
                    if "sound :" in line:
                        percent = line.strip().split(" ")[2]
                        res_map["sound"] = percent
                    elif "music :" in line:
                        percent = line.strip().split(" ")[2]
                        res_map["music"] = percent
                    elif "speech :" in line:
                        percent = line.strip().split(" ")[2]
                        res_map["speech"] = percent
                    elif "Total Accuracy:" in line:
                        percent = line.strip().split(" ")[2]
                        res_map["Total Accuracy"] = percent
            res_list.append(res_map)
    res_list = sorted(res_list, key=lambda x: x["Total Accuracy"], reverse=True)
    header = res_list[0].keys()
    rows = [x.values() for x in res_list]
    print(tabulate.tabulate(rows, header))


show_acc()
