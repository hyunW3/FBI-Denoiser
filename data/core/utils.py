import os
def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%.2f %s" % (s, size_name[i])
def get_number_from_MB(converted_size : str)-> float:
    return float(converted_size.split(" ")[0])
def show_file_info():
    data_path = "data/Samsung_SNU"
    for set_num in sorted(os.listdir(data_path)):
        set_path = os.path.join(data_path,set_num)
        cnt = 0

        for f_num in sorted(os.listdir(set_path)):
            f_path = os.path.join(set_path,f_num)
            avg_info = {"lowest_size" : -1, "mean_size" : 0, "cnt" : 0, "highest_size" : -1}
            for file_name in os.listdir(f_path):
                file_path = os.path.join(f_path,file_name)
                file_size = os.path.getsize(file_path)
                #print(file_name,file_size)

                if file_size == 0:
                    print(f"{file_name} has 0 bytes")
                else :
                    avg_info["cnt"] +=1
                    formatted_size = convert_size(file_size)
                    avg_info["mean_size"] += get_number_from_MB(formatted_size)
                    if avg_info["lowest_size"] == -1 or get_number_from_MB(formatted_size) < get_number_from_MB(avg_info["lowest_size"]):
                        avg_info["lowest_size"] = formatted_size
                    elif avg_info["highest_size"] == -1 or get_number_from_MB(formatted_size) > get_number_from_MB(avg_info["highest_size"]):
                        avg_info["highest_size"] = formatted_size
            avg_info["mean_size"] = str(round(avg_info["mean_size"] / avg_info["cnt"],2)) + " MB"
            print(set_num,f_num,"\t",avg_info)