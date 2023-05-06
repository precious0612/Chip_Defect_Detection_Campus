import os
import shutil

pic_path="/Users/precious/defect_detecting/data/intel/train/"
# pic_json_path="/Users/precious/defect_detecting/data/intel/train_json/"
temp_path="/Users/precious/defect_detecting/data/temp"
target_path="/Users/precious/defect_detecting/data/intel_target"
dirs = os.listdir(pic_path)

for item in dirs:
    file = os.path.join(pic_path,item)
    name = os.path.splitext(item)[0]
    json_file = pic_path + name + ".json"
    os.system("labelme " + file)
    # os.system("labelme_draw_json " + json_file)
    if os.path.exists(os.path.join(target_path,name+".png")):
        continue
    os.system("labelme_json_to_dataset " + json_file + " -o " + temp_path)
    shutil.copy(os.path.join(temp_path,"label.png"),os.path.join(target_path,name+".png"))