from collections import Counter
import tensorflow as tf

name_file_path = "data/raw/names/*.txt"


def frequent_name_extract(name_file_path):
    name_dic = dict()

    name_files = tf.gfile.Glob(name_file_path)

    if not name_files:
        print("Empty name files!")
        return

    for name_file in name_files:
        try:
            with open(name_file, "r") as f:
                contents = f.readlines()
        except:
            print("Broken name file: {0}".format(name_file))
            continue

        # in case a name is suitable for both sex
        for name_info in contents:
            name, sex, count = name_info.split(',')
            count = int(count)
            name_dic.setdefault(name, {'F': 0, 'M': 0})
            # print(name_dic[name])
            name_dic[name][sex] += count

    # choose major sex
    new_dict = {}
    for key in name_dic:
        if name_dic[key]['F'] > name_dic[key]['M']:
            new_dict[key] = {'count': name_dic[key]['F'], 'sex': 'female'}
        else:
            new_dict[key] = {'count': name_dic[key]['M'], 'sex': 'male'}

    # filter the names according to the paper
    name_dic = {k: v['sex'] for k, v in new_dict.items() if v['count'] >= 10000}

    return name_dic

def test():
    name_dic = frequent_name_extract(name_file_path)
    print(name_dic)

if __name__ == "__main__":
    test()